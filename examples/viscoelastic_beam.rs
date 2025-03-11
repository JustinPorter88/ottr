use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

use faer::{
    col,
    prelude::{c64, SpSolver},
    sparse::solvers::Eigendecomposition,
    unzipped, zipped, Col, ColRef, Mat, Scale,
};

use itertools::izip;
use ottr::{
    elements::beams::Damping,
    external::add_beamdyn_blade,
    model::Model,
    util::{quat_as_rotation_vector, ColAsMatRef, Quat},
    vtk::beams_qps_as_vtk,
    node::Direction,
    state::State,
};

fn main() {

    // Settings
    let inp_dir = "examples/inputs";
    let n_cycles = 3.5; // Number of oscillations to simulate
    let rho_inf = 1.; // Numerical damping
    let max_iter = 20; // Max convergence iterations
    let time_step = 0.01; // Time step


    // let out_dir = "output/bar_sub";

    // Box Beam Example from SONATA Repo
    let out_dir = "output/box_beam";
    let bd_file = "Box_Beam_BeamDyn.dat";
    let blade_file = "Box_Beam_BeamDyn_Blade.dat";
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
        fx.subrows_mut(0, 1).col_iter_mut(),
        solver.elements.beams.qp.x0.subrows(0, 1).col_iter(),
        solver.elements.beams.qp.m_star.col_iter(),
    )
    .for_each(|(mut fx_col, radius, m_star_col)| {
        // Need to extract the quadrature point mass here.
        fx_col[0] = m_star_col[0]*radius[0] * rot_rad_s * rot_rad_s;
    });

    undamped_model.set_distributed_loads(fx.clone());
    model.set_distributed_loads(fx.clone());

    // static state and solvers
    let mut static_state = undamped_model.create_state();
    let mut solver = undamped_model.create_solver();

    // // Point load:
    // // Get DOF index for beam tip node X direction and apply load
    // let tip_node_id = *node_ids.last().unwrap();
    // let tip_x_dof = solver.nfm.get_dof(tip_node_id, Direction::X).unwrap();
    // solver.fx[tip_x_dof] = 0.0e6; //1.0e6 gives a clear freq. shift.

    // println!("Mass[0,0] {:?}", solver.elements.beams.qp.m_star.col(0).subrows(0, 1));
    // println!("x0: {:?}", solver.elements.beams.qp.x0);
    println!("qp.fx (main): {:?}", solver.elements.beams.qp.fx);

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

    let (eig_val, _eig_vec) = modal_analysis(&out_dir, &undamped_model, base_state);

    let omega = Col::<f64>::from_fn(eig_val.nrows(), |i| eig_val[i].sqrt());

    println!("Frequency [Hz]: {:?}", Scale(1./2./PI) * &omega.subrows(0, 6));

    println!("Prestressed Modal Analysis:");

    let (eig_val, eig_vec) = modal_analysis(&out_dir, &undamped_model, static_state.clone());

    let omega = Col::<f64>::from_fn(eig_val.nrows(), |i| eig_val[i].sqrt());

    println!("Frequency [Hz]: {:?}", Scale(1./2./PI) * &omega.subrows(0, 6));

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

fn modal_analysis(out_dir: &str, model: &Model, mut state: State) -> (Col<f64>, Mat<f64>) {
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
    let stiff = solver.kt.clone().to_owned();

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

    (eig_val, eig_vec)
}
