use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

use faer::{
    col,
    prelude::{c64, SpSolver},
    sparse::solvers::Eigendecomposition,
    unzipped, zipped, Col, ColRef, Mat, MatRef, Scale
};

use itertools::{izip, Itertools};
use ottr::{
    elements::beams::Damping,
    external::add_beamdyn_blade,
    model::Model,
    util::{quat_as_rotation_vector, ColAsMatRef, Quat},
    vtk::beams_qps_as_vtk,
    node::Direction,
    state::State,
    elements::kernels::rotate_col_to_sectional
};

fn main() {

    // Settings
    let inp_dir = "examples/inputs";
    let n_cycles = 3.5; // Number of oscillations to simulate
    let rho_inf = 1.; // Numerical damping
    let max_iter = 20; // Max convergence iterations
    let time_step = 0.01; // Time step
    let nqp = Some(12); // Number of guass quad points to use. None-> trapezoid rule

    // see clear axial quadratic forces, so keep amplitudes small
    let tip_amp = 0.00007850090768953785;

    // let out_dir = "output/bar_sub";

    // Box Beam Example from SONATA Repo
    let out_dir = "output/box_beam";
    let bd_file = "Box_Beam_BeamDyn.dat";
    let blade_file = "Box_Beam_BeamDyn_Blade.dat";
    // let blade_file = "Box_Beam_BeamDyn_Blade_Clean.dat";
    let viscoelastic_file = "Box_Beam_BeamDyn_Blade_Viscoelastic.dat";

    // // 13 Thermoplastic Blade
    // let out_dir = "output/tp13m";
    // let blade_file = "Mass_BD_Blade.dat";
    // let bd_file = "IACMI_13m_thermoplastic_BeamDyn.dat";

    // ----- Model Setup ----------------------------------

    // Create output directory
    fs::create_dir_all(out_dir).unwrap();

    // Create undamped model and set solver parameters
    let mut undamped_model = Model::new();
    undamped_model.set_rho_inf(rho_inf);
    undamped_model.set_max_iter(max_iter);
    undamped_model.set_time_step(time_step);

    let undamped = Damping::None;

    // Add BeamDyn blade to model
    let (node_ids_undamped, _beam_elem_id) = add_beamdyn_blade(
        &mut undamped_model,
        // &format!("{inp_dir}/bar-subcomponent_BeamDyn.dat"),
        // &format!("{inp_dir}/bar-subcomponent_BeamDyn_Blade.dat"),
        &format!("{inp_dir}/{bd_file}"),
        &format!("{inp_dir}/{blade_file}"),
        10,
        undamped,
        None,
        nqp,
    );

    // Damped Model Creation
    let mut model = Model::new();
    model.set_rho_inf(rho_inf);
    model.set_max_iter(max_iter);
    model.set_time_step(time_step);

    let tmp_damping = Damping::None;

    let (node_ids, _beam_elem_id) = add_beamdyn_blade(
        &mut model,
        // &format!("{inp_dir}/bar-subcomponent_BeamDyn.dat"),
        // &format!("{inp_dir}/bar-subcomponent_BeamDyn_Blade.dat"),
        &format!("{inp_dir}/{bd_file}"),
        &format!("{inp_dir}/{blade_file}"),
        10,
        tmp_damping,
        Some(&format!("{inp_dir}/{viscoelastic_file}")),
        nqp,
    );

    // Prescribed constraint to first node of beam
    undamped_model.add_prescribed_constraint(node_ids_undamped[0]);
    model.add_prescribed_constraint(node_ids[0]);

    // ----- Static Analysis ----------------------------------

    undamped_model.set_static_solve();

    // create a solver so can access data on solver, but will
    // eventually recreate with the distrubted load.
    let mut solver = undamped_model.create_solver();

    // // Distributed load
    // solver.elements.beams.qp.fx
    //     .subrows_mut(0, 1)
    //     .col_iter_mut()
    //     .for_each(|mut fx| fx[0] = 1.0e4); //1e8 is good when have sufficient integration.

    // Variable Distributed load
    let n_qps = solver.elements.beams.qp.fx.ncols();
    let mut fx = Mat::<f64>::zeros(6, n_qps);

    let rot_rad_s = 6.0; // rad/s
    izip!(
        fx.subrows_mut(0, 1).col_iter_mut(), //xdof is row 0
        solver.elements.beams.qp.x0.subrows(0, 1).col_iter(),
        solver.elements.beams.qp.m_star.col_iter(),
    )
    .for_each(|(mut fx_col, radius, m_star_col)| {
        // Need to extract the quadrature point mass here.
        fx_col[0] = m_star_col[0]*radius[0] * rot_rad_s * rot_rad_s;
    });
    match nqp {
        Some(_) => println!("Cannot use this as distributed loads if have non-constant cross section!"),
        None => (),
    }
    // // Next two lines to actually set distribued loads
    // undamped_model.set_distributed_loads(fx.clone());
    // model.set_distributed_loads(fx.clone());

    // static state and solvers
    let mut static_state = undamped_model.create_state();
    let mut solver = undamped_model.create_solver();

    // Point load:
    // Get DOF index for beam tip node X direction and apply load
    let tip_node_id = *node_ids.last().unwrap();
    let tip_x_dof = solver.nfm.get_dof(tip_node_id, Direction::X).unwrap();
    solver.fx[tip_x_dof] = 0.0e3; //1.0e6 gives a clear freq. shift.

    // println!("Mass[0,0] {:?}", solver.elements.beams.qp.m_star.col(0).subrows(0, 1));
    // println!("x0: {:?}", solver.elements.beams.qp.x0);
    // println!("qp.fx (main): {:?}", solver.elements.beams.qp.fx);

    // Get static solution
    let _res = solver.step(&mut static_state);

    // Print displacements for reference
    let n_nodes = undamped_model.nodes.len();
    let s_nodes = Col::<f64>::from_fn(n_nodes, |i| undamped_model.nodes[i].s);

    println!("Static u_x: {:?}", static_state.u.row(0));
    // println!("Static x_x: {:?}", static_state.x.row(0)); // x = x_0 + u
    println!("Node positions on [0, 1]: {:?}", s_nodes);

    // ----- Eigen Analysis ----------------------------------

    model.set_dynamic_solve();

    // Perform modal analysis

    println!("Baseline Modal Analysis:");
    let mut base_state = undamped_model.create_state();

    let (eig_val, _eig_vec, mass_norm_amp_base) = modal_analysis(&out_dir, &undamped_model, base_state);

    let omega = Col::<f64>::from_fn(eig_val.nrows(), |i| eig_val[i].sqrt());

    println!("Frequency [Hz]: {:?}", Scale(1./2./PI) * &omega.subrows(0, 6));

    println!("Prestressed Modal Analysis:");

    let (eig_val, eig_vec, mass_norm_amp) = modal_analysis(&out_dir, &undamped_model, static_state.clone());

    let omega = Col::<f64>::from_fn(eig_val.nrows(), |i| eig_val[i].sqrt());

    println!("Frequency [Hz]: {:?}", Scale(1./2./PI) * &omega.subrows(0, 6));

    // ----- Static Analysis of Internal Modal Forces ---------------------
    // Only consider the undamped_model for this analysis

    // Apply only the eig_vec as a set of displacements
    let mode_ind=0;
    internal_forces(
        &undamped_model,
        &(static_state.clone()),
        eig_vec.col(0).clone(),
        out_dir,
        mode_ind+1,
        tip_amp,
        mass_norm_amp_base[mode_ind],
        mass_norm_amp[mode_ind],
    );

    // println!("Tip amplitude for unit modal: {:?}", 1./mass_norm_amp[0]);

    // ----- Transient Time Integration ----------------------------------

    // Loop through modes and run simulation
    izip!(omega.iter(), eig_vec.col_iter())
        .take(4)
        .enumerate()
        .for_each(|(i, (&omega, shape))| {
            let t_end = 2. * PI / omega;
            let time_step = t_end / 100.;
            let n_steps = (n_cycles * t_end / time_step) as usize;

            let mut curr_model = model.clone();
            curr_model.set_time_step(time_step);

            // run_simulation(i + 1, time_step, n_steps, shape, out_dir, model.clone());
            run_simulation(i + 1, time_step, n_steps, shape, out_dir, curr_model);
        });
}

fn run_simulation(
    mode: usize,
    time_step: f64,
    n_steps: usize,
    shape: ColRef<f64>,
    out_dir: &str,
    mut model: Model,
) {
    // Create new solver where beam elements have damping
    model.set_static_solve();
    let mut solver = model.create_solver();
    let mut static_state = model.create_state();

    // Do static solve for inital displacements
    solver.step(&mut static_state);

    // Dynamic states/solver
    model.set_dynamic_solve();
    let mut solver = model.create_solver();
    let mut state = static_state.clone();

    // Apply scaled mode shape to state as velocity
    let v = shape;
    state.v.copy_from(v.as_ref().as_mat_ref(6, state.n_nodes));

    // Create output file
    let mut file = File::create(format!("{out_dir}/displacement_{:02}.csv", mode)).unwrap();

    // Cartesian rotation vector
    let mut rv = Col::<f64>::zeros(3);

    // Loop through times and run simulation
    for i in 0..n_steps {
        // Calculate time
        let t = (i as f64) * time_step;

        write!(file, "{t}").unwrap();
        state.u.col_iter().for_each(|c| {
            quat_as_rotation_vector(c.subrows(3, 4), rv.as_mut());
            write!(
                file,
                ",{},{},{},{},{},{}",
                c[0], c[1], c[2], rv[0], rv[1], rv[2]
            )
            .unwrap();
        });
        file.write(b"\n").unwrap();

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
        }

        assert_eq!(res.converged, true);
    }
}

fn write_fc_star(
    mut file: File,
    stations : &Vec<f64>,
    weights : &Vec<f64>,
    f_c: MatRef<f64>,
    mass_norm_amp: f64
    ){
    write!(
        file,
        "Mass normalized modal amplitude : {} \n", mass_norm_amp
    ).unwrap();
    write!(
        file,
        "station,weight,Fx,Fy,Fz,Mx,My,Mz (OpenTurbine Coordinates) \n",
    ).unwrap();

    izip!(stations.iter(), weights.iter(), f_c.col_iter())
        .for_each(|(stat, w, f_c_col)|
    {
        write!(
            file,
            "{},{},{},{},{},{},{},{}\n",
            stat,w,f_c_col[0], f_c_col[1], f_c_col[2], f_c_col[3], f_c_col[4], f_c_col[5]
        ).unwrap();
    });
}

fn internal_forces(
    model: &Model,
    static_state: &State,
    eigen_shape: ColRef<f64>,
    out_dir: &str,
    mode: usize,
    tip_amp : f64,
    shape_to_mass_norm: f64, // modal w/o prestress
    shape_to_mass_norm_prestress: f64 // prestressed modal
) {
    // Only consider the undamped_model for this analysis


    // Apply only the static displacements
    let mut solver_undamped = model.create_solver();

    solver_undamped.elements.assemble_system(
        static_state,
        &solver_undamped.nfm,
        solver_undamped.p.h,
        solver_undamped.m.as_mut(),
        solver_undamped.ct.as_mut(),
        solver_undamped.kt.as_mut(),
        solver_undamped.r.as_mut(),
    );

    let nqp = solver_undamped.elements.beams.qp.fe_c.shape().1;
    let mut fe_c_star = Mat::<f64>::zeros(6, nqp);

    rotate_col_to_sectional(
        fe_c_star.as_mut(),
        solver_undamped.elements.beams.qp.fe_c.as_ref(),
        solver_undamped.elements.beams.qp.rr0.as_ref()
    );

    println!("fe_c_star (static) : {:?}", fe_c_star);

    // let section_loc = model.beam_elements[0].sections.iter().map(|s| s.s).collect_vec();
    let section_loc = model.beam_elements[0].quadrature.points.iter().map(|&s| (s+1.)/2.).collect_vec();
    let section_weights = model.beam_elements[0].quadrature.weights.iter().map(|&w| w).collect_vec();


    write_fc_star(
        File::create(format!("{out_dir}/fc_star_prestress_{:02}.csv", mode)).unwrap(),
        &section_loc,
        &section_weights,
        fe_c_star.as_ref(),
        0.0
    );


    // Apply only the eig_vec as a set of displacements
    let mut eig_state = model.create_state();
    let h = 1.;
    let u = eigen_shape * Scale(tip_amp);

    eig_state.u_prev.fill_zero();
    eig_state.u_prev.row_mut(3).fill(1.);
    eig_state.u_delta.copy_from(&u.as_ref().as_mat_ref(6, eig_state.n_nodes));
    eig_state.calc_displacement(h);
    eig_state.calculate_x();

    println!("Eigen state.u {:?}", eig_state.u);

    solver_undamped.elements.assemble_system(
        &eig_state,
        &solver_undamped.nfm,
        solver_undamped.p.h,
        solver_undamped.m.as_mut(),
        solver_undamped.ct.as_mut(),
        solver_undamped.kt.as_mut(),
        solver_undamped.r.as_mut(),
    );

    let mut fe_c_star = Mat::<f64>::zeros(6, nqp);

    rotate_col_to_sectional(
        fe_c_star.as_mut(),
        solver_undamped.elements.beams.qp.fe_c.as_ref(),
        solver_undamped.elements.beams.qp.rr0.as_ref()
    );


    write_fc_star(
        File::create(format!("{out_dir}/fc_star_eigen_{:02}.csv", mode)).unwrap(),
        &section_loc,
        &section_weights,
        fe_c_star.as_ref(),
        shape_to_mass_norm * tip_amp
    );


    // Apply both combined - need to verify how to do this.


    // Apply only the eig_vec as a set of displacements
    let mut eig_pre_state = static_state.clone();
    let h = 1.;
    let u = eigen_shape * Scale(tip_amp);

    eig_pre_state.u_prev.copy_from(eig_pre_state.u.clone());
    eig_pre_state.u_delta.copy_from(&u.as_ref().as_mat_ref(6, eig_state.n_nodes));
    eig_pre_state.calc_displacement(h);
    eig_pre_state.calculate_x();

    println!("Pre + Eigen state.u {:?}", eig_state.u);

    solver_undamped.elements.assemble_system(
        &eig_pre_state,
        &solver_undamped.nfm,
        solver_undamped.p.h,
        solver_undamped.m.as_mut(),
        solver_undamped.ct.as_mut(),
        solver_undamped.kt.as_mut(),
        solver_undamped.r.as_mut(),
    );

    let mut fe_c_star = Mat::<f64>::zeros(6, nqp);

    rotate_col_to_sectional(
        fe_c_star.as_mut(),
        solver_undamped.elements.beams.qp.fe_c.as_ref(),
        solver_undamped.elements.beams.qp.rr0.as_ref()
    );


    write_fc_star(
        File::create(format!("{out_dir}/fc_star_pre_and_eigen_{:02}.csv", mode)).unwrap(),
        &section_loc,
        &section_weights,
        fe_c_star.as_ref(),
        shape_to_mass_norm_prestress * tip_amp
    );

}

fn modal_analysis(out_dir: &str, model: &Model, mut state: State) -> (Col<f64>, Mat<f64>, Col<f64>) {
    // Create solver and state from model
    let mut solver = model.create_solver();

    // should not matter in modal analysis, argument needed for viscoelastic
    let h = 0.0;

    // Calculate system based on initial state
    solver.elements.beams.calculate_system(&state, h);

    // Get matrices
    solver.elements.beams.assemble_system(
        &solver.nfm,
        solver.m.as_mut(),
        solver.ct.as_mut(),
        solver.kt.as_mut(),
        solver.r.as_mut(),
    );

    let mass: Mat<f64> = solver.m.clone().to_owned();
    // let stiff = solver.kt.clone().to_owned();

    println!("This needs to be updated for free-free");
    let ndof_bc = solver.n_system - 6;
    let lu = solver.m.submatrix(6, 6, ndof_bc, ndof_bc).partial_piv_lu();
    let a = lu.solve(solver.kt.submatrix(6, 6, ndof_bc, ndof_bc));

    let eig: Eigendecomposition<c64> = a.eigendecomposition();
    let eig_val_raw = eig.s().column_vector();
    let eig_vec_raw = eig.u();

    let mut eig_order: Vec<_> = (0..eig_val_raw.nrows()).collect();
    eig_order.sort_by(|&i, &j| {
        eig_val_raw
            .get(i)
            .re
            .partial_cmp(&eig_val_raw.get(j).re)
            .unwrap()
    });

    let eig_val = Col::<f64>::from_fn(eig_val_raw.nrows(), |i| eig_val_raw[eig_order[i]].re);
    let mut eig_vec = Mat::<f64>::from_fn(solver.n_system, eig_vec_raw.ncols(), |i, j| {
        if i < 6 {
            0.
        } else {
            eig_vec_raw[(i - 6, eig_order[j])].re
        }
    });
    // normalize eigen vectors
    eig_vec.as_mut().col_iter_mut().for_each(|mut c| {
        let max = *c
            .as_ref()
            .iter()
            .reduce(|acc, e| if e.abs() > acc.abs() { e } else { acc })
            .unwrap();
        zipped!(&mut c).for_each(|unzipped!(c)| *c /= max);
    });

    // Calculate the mass normalized modal amplitude for each of the mode shapes
    let mut mass_norm_amp = Col::<f64>::zeros(eig_vec.shape().1);

    eig_vec.col_iter().enumerate().for_each(|(i, phi)|{
        mass_norm_amp[i] = (phi.transpose() * &mass * phi).sqrt();
    });

    // println!("phi^T M phi: {:?}", eig_vec.clone().transpose() * &mass * eig_vec.clone());

    // Write eigenanalysis results to output file
    let mut file = File::create(format!("{out_dir}/compare_eigenanalysis.csv")).unwrap();
    write!(file, "freq").unwrap();
    for i in 1..=state.n_nodes {
        for d in ["u1", "u2", "u3", "r1", "r2", "r3"] {
            write!(file, ",N{i}_{d}").unwrap();
        }
    }
    izip!(eig_val.iter(), eig_vec.col_iter()).for_each(|(&lambda, c)| {
        write!(file, "\n{}", lambda.sqrt() / (2. * PI)).unwrap();
        for &v in c.iter() {
            write!(file, ",{v}").unwrap();
        }
    });

    // Write mode shapes to output file
    let mut file = File::create(format!("{out_dir}/compare_modes.csv")).unwrap();
    write!(file, "freq").unwrap();
    for i in 1..=solver.elements.beams.qp.x.ncols() {
        for d in ["u1", "u2", "u3", "r1", "r2", "r3"] {
            write!(file, ",qp{i}_{d}").unwrap();
        }
    }

    fs::create_dir_all(format!("{out_dir}/vtk")).unwrap();
    let mut q = col![0., 0., 0., 0.];
    let mut rv = col![0., 0., 0.];
    izip!(0..eig_val.nrows(), eig_val.iter(), eig_vec.col_iter()).for_each(
        |(i, &lambda, phi_col)| {
            // Apply eigvector displacements to state
            let phi = phi_col.as_mat_ref(6, state.u.ncols());

            izip!(state.u.col_iter_mut(), phi.col_iter()).for_each(|(mut u, phi)| {
                let phi = phi * Scale(1.);
                q.as_mut().quat_from_rotation_vector(phi.subrows(3, 3));
                u[0] = phi[0];
                u[1] = phi[1];
                u[2] = phi[2];
                u[3] = q[0];
                u[4] = q[1];
                u[5] = q[2];
                u[6] = q[3];
            });

            // should not matter in modal analysis, argument needed for viscoelastic
            let h = 0.0;

            // Update beam elements from state
            solver.elements.beams.calculate_system(&state, h);

            // Write frequency to file
            write!(file, "\n{}", lambda.sqrt() / (2. * PI)).unwrap();

            // Loop through element quadrature points and output position and rotation
            for c in solver.elements.beams.qp.u.col_iter() {
                quat_as_rotation_vector(c.subrows(3, 4), rv.as_mut());
                write!(
                    file,
                    ",{},{},{},{},{},{}",
                    c[0], c[1], c[2], rv[0], rv[1], rv[2]
                )
                .unwrap();
            }

            beams_qps_as_vtk(&solver.elements.beams)
                .export_ascii(format!("{out_dir}/vtk/mode.{:0>3}.vtk", i + 1))
                .unwrap()
        },
    );

    (eig_val, eig_vec, mass_norm_amp)
}
