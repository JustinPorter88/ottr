use core::panic;
use std::{f64::consts::PI, fs};

use faer::{col, prelude::SpSolver, Mat, Col};
use itertools::{izip, Itertools};

use crate::{
    elements::beams::{BeamSection, Damping},
    interp::{gauss_legendre_lobotto_points, shape_deriv_matrix, shape_interp_matrix},
    model::Model,
    quadrature::Quadrature,
    util::{quat_as_matrix, ColAsMatMut, Quat, ColAsMatRef},
};

pub fn add_beamdyn_blade(
    model: &mut Model,
    bd_primary_file_path: &str,
    bd_blade_file_path: &str,
    elem_order: usize,
    mut damping: Damping,
    viscoelastic_file_path: Option<&str>,
) -> (Vec<usize>, usize) {
    // Read key points
    let key_points = parse_beamdyn_primary_file(&fs::read_to_string(bd_primary_file_path).unwrap());

    // Read sections
    let mut sections = parse_beamdyn_blade_file(&fs::read_to_string(bd_blade_file_path).unwrap());

    let mut new_damping = Damping::None;
    match viscoelastic_file_path {
        Some(path) => {
            let (nqp_visco, damping_tmp) = parse_beamdyn_viscoelastic_file(&fs::read_to_string(path).unwrap());
            let mut c_star_visco = Mat::<f64>::zeros(36, nqp_visco);

            if !(nqp_visco == sections.len()) {
                panic!("Number of visoelastic stations does not match number of quadrature points.")
            }

            match damping_tmp {
                Damping::Viscoelastic(ctau_star, tau_i) =>{
                    let nrows = tau_i.nrows() - 1;
                    let ctau_star_sub = ctau_star.subrows(0, 36*nrows).to_owned();
                    let tau_i_sub = tau_i.subrows(0, nrows).to_owned();

                    new_damping = Damping::Viscoelastic(ctau_star_sub, tau_i_sub);

                    c_star_visco.copy_from(ctau_star.subrows(36*nrows, 36));
                },
                _ => ()
            }

            // Recreate sections but use c_star_visco for the c_star field.
            let updated_sections = sections
                .iter()
                .enumerate()
                .map(|(ind, s)|
                    BeamSection {
                        s : s.s,
                        m_star: s.m_star.clone(),
                        c_star: c_star_visco.col(ind).as_mat_ref(6, 6).clone().to_owned(),
                    }
                ).collect_vec();

            damping = new_damping;
            sections = updated_sections;

        },
        None => (),
    }

    // Calculate key point position on range of [-1,1]
    let n_kps = key_points.len();
    let kp_range = key_points[n_kps - 1][0] - key_points[0][0];
    let kp_xi = key_points
        .iter()
        .map(|kp| 2. * (kp[0] - key_points[0][0]) / kp_range - 1.)
        .collect_vec();

    // Get node positions on [-1,1] and [0,1]
    let n_nodes = elem_order + 1;
    let node_xi = gauss_legendre_lobotto_points(elem_order);
    let node_s = node_xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

    // Build interpolation matrix to go from key points to node
    let mut shape_interp = Mat::<f64>::zeros(kp_xi.len(), node_xi.len());
    shape_interp_matrix(&kp_xi, &node_xi, shape_interp.as_mut());

    // Build A matrix for fitting key points to element nodes
    let mut a_matrix = Mat::<f64>::zeros(n_nodes, n_nodes);
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            for k in 0..n_kps {
                a_matrix[(i, j)] += shape_interp[(k, i)] * shape_interp[(k, j)];
            }
        }
    }
    a_matrix.row_mut(0).fill(0.);
    a_matrix.row_mut(n_nodes - 1).fill(0.);
    a_matrix[(0, 0)] = 1.;
    a_matrix[(n_nodes - 1, n_nodes - 1)] = 1.;

    // Build B matrix for fitting key points to element nodes
    let kp_matrix = Mat::from_fn(n_kps, 4, |i, j| key_points[i][j]);
    let mut b_matrix = shape_interp.transpose() * &kp_matrix;
    b_matrix.row_mut(0).copy_from(&kp_matrix.row(0));
    b_matrix
        .row_mut(n_nodes - 1)
        .copy_from(&kp_matrix.row(n_kps - 1));

    // Solve for node locations using least squares
    let lu = a_matrix.full_piv_lu();
    let node_xyzt = lu.solve(&b_matrix);

    // Calculate derivative at nodes
    let mut shape_deriv = Mat::<f64>::zeros(node_xi.len(), node_xi.len());
    shape_deriv_matrix(&node_xi, &node_xi, shape_deriv.as_mut());
    let deriv = shape_deriv * &node_xyzt.subcols(0, 3);

    // Loop through node locations and derivatives and add to model
    let mut q = col![0., 0., 0., 0.];
    let node_ids = izip!(node_s.iter(), node_xyzt.row_iter(), deriv.row_iter())
        .map(|(&si, nd, deriv)| {
            let twist = nd[3];
            let tan = deriv.transpose() / deriv.norm_l2();
            q.as_mut().quat_from_tangent_twist(tan.as_ref(), twist); // Calculate twist about tangent
            model
                .add_node()
                .element_location(si)
                .position(nd[0], nd[1], nd[2], q[0], q[1], q[2], q[3])
                .build()
        })
        .collect_vec();

    // Quadrature rule
    let gq = Quadrature::trapezoidal(&sections.iter().map(|s| s.s).collect_vec());

    // Add beam element
    let beam_elem_id = model.add_beam_element(&node_ids, &gq, &sections, damping);

    (node_ids, beam_elem_id)
}

pub fn parse_beamdyn_primary_file(file_data: &str) -> Vec<[f64; 4]> {
    let lines = file_data.lines().collect_vec();

    let member_total_line = lines.get(19).unwrap();
    if !member_total_line.contains("member_total") {
        panic!("line 20 doesn't contain member_total")
    }
    let member_total: usize = member_total_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    if member_total != 1 {
        panic!("member_total must equal 1")
    }

    // Get the number of key points
    let kp_total_line = lines.get(20).unwrap();
    if !kp_total_line.contains("kp_total") {
        panic!("line 21 doesn't contain kp_total")
    }
    let kp_total = kp_total_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    // Get key point coordinates and twist (swap x and z so length is along x)
    lines
        .iter()
        .skip(24)
        .take(kp_total)
        .map(|l| {
            let v = l
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect_vec();
            [v[2], v[1], v[0], v[3]]
        })
        .collect_vec()
}

pub fn parse_beamdyn_blade_file(file_data: &str) -> Vec<BeamSection> {
    let mut m = Mat::<f64>::zeros(3, 3);
    let mut q_rot = col![0., 0., 0., 0.];
    q_rot
        .as_mut()
        .quat_from_rotation_vector(col![0., -PI / 2., 0.].as_ref());
    quat_as_matrix(q_rot.as_ref(), m.as_mut());
    let mut m_rot = Mat::<f64>::zeros(6, 6);
    m_rot.submatrix_mut(0, 0, 3, 3).copy_from(&m);
    m_rot.submatrix_mut(3, 3, 3, 3).copy_from(&m);
    let lines = file_data.lines().skip(10).collect_vec();
    lines
        .iter()
        .chunks(15)
        .into_iter()
        .map(|chunks| {
            let ls = chunks.collect_vec();
            let s = ls[0].trim().parse::<f64>().unwrap();
            let c = (1..7)
                .map(|i| {
                    ls[i]
                        .split_ascii_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect_vec()
                })
                .collect_vec();
            let m = (8..14)
                .map(|i| {
                    ls[i]
                        .split_ascii_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect_vec()
                })
                .collect_vec();
            BeamSection {
                s,
                m_star: &m_rot * Mat::<f64>::from_fn(6, 6, |i, j| m[i][j]) * &m_rot.transpose(),
                c_star: &m_rot * Mat::<f64>::from_fn(6, 6, |i, j| c[i][j]) * &m_rot.transpose(),
            }
        })
        .collect_vec()
}

pub fn parse_beamdyn_viscoelastic_file(file_data: &str) -> (usize, Damping) {
    let mut m = Mat::<f64>::zeros(3, 3);
    let mut q_rot = col![0., 0., 0., 0.];
    q_rot
        .as_mut()
        .quat_from_rotation_vector(col![0., -PI / 2., 0.].as_ref());
    quat_as_matrix(q_rot.as_ref(), m.as_mut());
    let mut m_rot = Mat::<f64>::zeros(6, 6);
    m_rot.submatrix_mut(0, 0, 3, 3).copy_from(&m);
    m_rot.submatrix_mut(3, 3, 3, 3).copy_from(&m);

    let lines: Vec<&str> = file_data.lines().collect_vec();

    // Get the number of stations
    let stations_line = lines.get(3).unwrap();
    if !stations_line.contains("station_total") {
        panic!("line 4 doesn't contain stations_total.")
    }
    let n_stations: usize = stations_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    // Get number of time scales
    let time_scales_line = lines.get(4).unwrap();
    if !time_scales_line.contains("time_scales_total") {
        panic!("line 5 doesn't contain kp_total")
    }
    let n_prony: usize = time_scales_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    // Get the time scales values
    let time_list_line = lines.get(6).unwrap();

    let tau_i_vec = time_list_line.split_ascii_whitespace()
                                .filter_map(|s| s.parse::<f64>().ok())
                                .collect_vec();


    let tau_i = Col::<f64>::from_fn(n_prony, |i| tau_i_vec[i]);
    let mut stations = Col::<f64>::zeros(n_stations);

    // println!("Num tau_i: {:?}, tau values: {:?}", n_prony, tau_i_vec);
    // println!("tau_i as col values: {:?}", tau_i);

    let mut ctau_star_all = Mat::<f64>::zeros(6*6*n_prony,n_stations);

    let lines = file_data.lines().skip(8).collect_vec();
    izip!(
        lines
            .iter()
            .chunks(1+7*n_prony)
            .into_iter(),
        ctau_star_all.col_iter_mut().enumerate(),
    ).for_each(|(chunks, (station_ind, ctau_star_station))| {

        let ls = chunks.collect_vec();

        // section fraction of blade length
        stations[station_ind] = ls[0].trim().parse::<f64>().unwrap();

        let ctau_star_mat = ctau_star_station.as_mat_mut(36, n_prony);

        // Inner loop of each time scale at this station
        ctau_star_mat
            .col_iter_mut()
            .enumerate()
            .for_each(|(time_ind, ctau_col)|
        {
            let c = (1+7*time_ind..7+7*time_ind)
                            .map(|i| {
                                ls[i]
                                    .split_ascii_whitespace()
                                    .filter_map(|s| s.parse::<f64>().ok())
                                    .collect_vec()
                            })
                            .collect_vec();

            // println!("c: {:?}", c);

            let c_star = &m_rot * Mat::<f64>::from_fn(6, 6, |i, j| c[i][j]) * &m_rot.transpose();

            let mut c_dest_mat = ctau_col.as_mat_mut(6, 6);
            c_dest_mat.copy_from(c_star);
        });

    });

    let damping = Damping::Viscoelastic(ctau_star_all, tau_i);

    (n_stations, damping)
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_beamdyn_keypoints() {
        let keypoints = parse_beamdyn_primary_file(BD_INPUT1);
        println!("{:?}", keypoints);
    }

    #[test]
    fn test_parse_beamdyn_blade_file() {
        let sections = parse_beamdyn_blade_file(BD_INPUT2);
        println!("{:?}", sections);
    }

    const BD_INPUT1: &str = r#"--------- BEAMDYN with OpenFAST INPUT FILE -------------------------------------------
bar_urc blade
---------------------- SIMULATION CONTROL --------------------------------------
True          Echo             - Echo input data to "<RootName>.ech"? (flag)
False         QuasiStaticInit  - Use quasi-static pre-conditioning with centripetal accelerations in initialization? (flag) [dynamic solve only]
          0   rhoinf           - Numerical damping parameter for generalized-alpha integrator
          1   quadrature       - Quadrature method: 1=Gaussian; 2=Trapezoidal (switch)
          1   refine           - Refinement factor for trapezoidal quadrature (-) [DEFAULT = 1; used only when quadrature=2]
"DEFAULT"     n_fact           - Factorization frequency for the Jacobian in N-R iteration(-) [DEFAULT = 5]
"DEFAULT"     DTBeam           - Time step size (s)
50     load_retries     - Number of factored load retries before quitting the simulation [DEFAULT = 20]
"DEFAULT"     NRMax            - Max number of iterations in Newton-Raphson algorithm (-) [DEFAULT = 10]
"DEFAULT"     stop_tol         - Tolerance for stopping criterion (-) [DEFAULT = 1E-5]
"DEFAULT"     tngt_stf_fd      - Use finite differenced tangent stiffness matrix? (flag)
"DEFAULT"     tngt_stf_comp    - Compare analytical finite differenced tangent stiffness matrix? (flag)
"DEFAULT"     tngt_stf_pert    - Perturbation size for finite differencing (-) [DEFAULT = 1E-6]
"DEFAULT"     tngt_stf_difftol - Maximum allowable relative difference between analytical and fd tangent stiffness (-); [DEFAULT = 0.1]
True          RotStates        - Orient states in the rotating frame during linearization? (flag) [used only when linearizing]
---------------------- GEOMETRY PARAMETER --------------------------------------
          1   member_total    - Total number of members (-)
          5   kp_total        - Total number of key points (-) [must be at least 3]
     1     5                  - Member number; Number of key points in this member
		 kp_xr 			 kp_yr 			 kp_zr 		 initial_twist
		  (m)  			  (m)  			  (m)  		   (deg)
	 0.00000e+00 	 0.00000e+00 	 0.00000e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 2.04082e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 4.08163e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 6.12245e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 8.16327e+00 	 0.00000e+00 "#;

    const BD_INPUT2: &str =
        "------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------
! NACA 0012 airfoil with chord 0.1 - Written using beamdyn.py
---------------------- BLADE PARAMETERS --------------------------------------
   21  station_total    - Number of blade input stations (-)
    1  damp_type        - Damping type (switch): 0: no damping; 1: viscous damping
---------------------- DAMPING COEFFICIENT------------------------------------
   mu1        mu2        mu3        mu4        mu5        mu6
   (s)        (s)        (s)        (s)        (s)        (s)
 0.01 0.01 0.01 0.01 0.01 0.01
---------------------- DISTRIBUTED PROPERTIES---------------------------------
0.000000
   6.804663E-18 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 -1.549113E-28
   0.000000E+00 5.440283E-18 0.000000E+00 0.000000E+00 0.000000E+00 6.234980E-20
   0.000000E+00 0.000000E+00 1.715755E+08 -8.494318E-11 -2.880672E+06 0.000000E+00
   0.000000E+00 0.000000E+00 -8.494318E-11 1.134880E+05 -5.616727E+04 0.000000E+00
   0.000000E+00 0.000000E+00 -2.880672E+06 -5.616727E+04 7.793935E+04 0.000000E+00
   -1.549113E-28 6.234980E-20 0.000000E+00 0.000000E+00 0.000000E+00 2.120621E+03

   6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 3.175257E-18
   0.000000E+00 6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 1.076823E-01
   0.000000E+00 0.000000E+00 6.413657E+00 -3.175257E-18 -1.076823E-01 0.000000E+00
   0.000000E+00 0.000000E+00 -3.175257E-18 6.776322E-09 5.331106E-20 0.000000E+00
   0.000000E+00 0.000000E+00 -1.076823E-01 5.331106E-20 1.808609E-03 0.000000E+00
   3.175257E-18 1.076823E-01 0.000000E+00 0.000000E+00 0.000000E+00 1.808616E-03

0.050000
   6.804663E-18 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 -1.549113E-28
   0.000000E+00 5.440283E-18 0.000000E+00 0.000000E+00 0.000000E+00 6.234980E-20
   0.000000E+00 0.000000E+00 1.715755E+08 -8.494318E-11 -2.880672E+06 0.000000E+00
   0.000000E+00 0.000000E+00 -8.494318E-11 1.134880E+05 -5.616727E+04 0.000000E+00
   0.000000E+00 0.000000E+00 -2.880672E+06 -5.616727E+04 7.793935E+04 0.000000E+00
   -1.549113E-28 6.234980E-20 0.000000E+00 0.000000E+00 0.000000E+00 2.120621E+03

   6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 3.175257E-18
   0.000000E+00 6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 1.076823E-01
   0.000000E+00 0.000000E+00 6.413657E+00 -3.175257E-18 -1.076823E-01 0.000000E+00
   0.000000E+00 0.000000E+00 -3.175257E-18 6.776322E-09 5.331106E-20 0.000000E+00
   0.000000E+00 0.000000E+00 -1.076823E-01 5.331106E-20 1.808609E-03 0.000000E+00
   3.175257E-18 1.076823E-01 0.000000E+00 0.000000E+00 0.000000E+00 1.808616E-03

0.100000
   6.804663E-18 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 -1.549113E-28
   0.000000E+00 5.440283E-18 0.000000E+00 0.000000E+00 0.000000E+00 6.234980E-20
   0.000000E+00 0.000000E+00 1.715755E+08 -8.494318E-11 -2.880672E+06 0.000000E+00
   0.000000E+00 0.000000E+00 -8.494318E-11 1.134880E+05 -5.616727E+04 0.000000E+00
   0.000000E+00 0.000000E+00 -2.880672E+06 -5.616727E+04 7.793935E+04 0.000000E+00
   -1.549113E-28 6.234980E-20 0.000000E+00 0.000000E+00 0.000000E+00 2.120621E+03

   6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 3.175257E-18
   0.000000E+00 6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 1.076823E-01
   0.000000E+00 0.000000E+00 6.413657E+00 -3.175257E-18 -1.076823E-01 0.000000E+00
   0.000000E+00 0.000000E+00 -3.175257E-18 6.776322E-09 5.331106E-20 0.000000E+00
   0.000000E+00 0.000000E+00 -1.076823E-01 5.331106E-20 1.808609E-03 0.000000E+00
   3.175257E-18 1.076823E-01 0.000000E+00 0.000000E+00 0.000000E+00 1.808616E-03";
}
