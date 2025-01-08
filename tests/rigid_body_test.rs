use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

use faer::{assert_matrix_eq, col, mat, solvers::SpSolver, Col, Mat, Scale};
use itertools::Itertools;
use ottr::{
    interfaces::{MooringLine, RigidPlatform},
    model::Model,
    node::Direction,
    util::{cross, quat_as_euler_angles, quat_as_rotation_vector, vec_tilde},
    vtk::lines_as_vtk,
};

#[test]
fn test_precession() {
    // Model
    let mut model = Model::new();

    let mass_node = model
        .add_node()
        .position(0., 0., 0., 1., 0., 0., 0.)
        .velocity(0., 0., 0., 0.5, 0.5, 1.)
        .build();

    // Define mass matrix
    let m = 1.;
    let mut mass_matrix = Mat::<f64>::zeros(6, 6);
    mass_matrix
        .diagonal_mut()
        .column_vector_mut()
        .copy_from(col![m, m, m, 1., 1., 0.5]);

    // Add mass element
    model.add_mass_element(mass_node, mass_matrix);

    //--------------------------------------------------------------------------
    // Create state and solver
    //--------------------------------------------------------------------------

    let mut state = model.create_state();

    let time_step = 0.01;
    model.set_time_step(time_step);
    model.set_rho_inf(1.);
    model.set_max_iter(6);
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // run simulation
    //--------------------------------------------------------------------------

    for _ in 0..500 {
        // Output current rotation
        // let q = state.u.col(0).subrows(3, 4);
        // quat_as_euler_angles(q, e.as_mut());
        // println!("{}\t{}\t{}\t{}", (i as f64) * time_step, e[0], e[1], e[2]);

        // Step
        let res = solver.step(&mut state);
        assert_eq!(res.converged, true);
    }

    let mut e = Col::<f64>::zeros(3);
    quat_as_euler_angles(state.u.col(0).subrows(3, 4), e.as_mut());

    assert_matrix_eq!(
        e.as_2d(),
        col![-1.413542763236864, 0.999382175365794, 0.213492011335111].as_2d(),
        comp = float
    )
}

#[test]
fn test_heavy_top() {
    let out_dir = "output";
    let time_step: f64 = 0.002;
    let t_end: f64 = 2.;
    let n_steps = ((t_end / time_step).ceil() as usize) + 1;

    // Model
    let mut model = Model::new();
    model.set_solver_tolerance(1e-5, 1.);
    model.set_time_step(time_step);
    model.set_rho_inf(0.9);
    model.set_max_iter(6);

    // Heavy top parameters
    let m = 15.;
    let mut j = Mat::<f64>::zeros(3, 3);
    j.diagonal_mut()
        .column_vector_mut()
        .copy_from(col![0.234375, 0.46875, 0.234375]);
    let omega = col![0., 150., -4.61538];
    let gamma = col![0., 0., -9.81];
    let x = col![0., 1., 0.];

    // translational velocity
    let mut x_dot = col![0., 0., 0.];
    cross(omega.as_ref(), x.as_ref(), x_dot.as_mut());

    // angular acceleration
    let mut x_tilde = Mat::<f64>::zeros(3, 3);
    vec_tilde(x.as_ref(), x_tilde.as_mut());
    let j_bar: Mat<f64> = &j - m * &x_tilde * &x_tilde;
    let mut x_cross_m_gamma = col![0., 0., 0.];
    cross(
        x.as_ref(),
        (Scale(m) * &gamma).as_ref(),
        x_cross_m_gamma.as_mut(),
    );
    let mut omega_cross_j_bar_omega = col![0., 0., 0.];
    let j_bar_omega: Col<f64> = &j_bar * &omega;
    cross(
        omega.as_ref(),
        j_bar_omega.as_ref(),
        omega_cross_j_bar_omega.as_mut(),
    );
    let omega_dot = j_bar
        .partial_piv_lu()
        .solve(&x_cross_m_gamma - &omega_cross_j_bar_omega);

    // translational acceleration
    let mut omega_dot_cross_x = col![0., 0., 0.];
    cross(omega_dot.as_ref(), x.as_ref(), omega_dot_cross_x.as_mut());
    let mut omega_cross_x_dot = col![0., 0., 0.];
    cross(omega.as_ref(), x_dot.as_ref(), omega_cross_x_dot.as_mut());
    let x_ddot = omega_dot_cross_x + omega_cross_x_dot;

    // Add mass element
    let mass_node_id = model
        .add_node()
        .position(x[0], x[1], x[2], 1., 0., 0., 0.)
        .velocity(x_dot[0], x_dot[1], x_dot[2], omega[0], omega[1], omega[2])
        .acceleration(
            x_ddot[0],
            x_ddot[1],
            x_ddot[2],
            omega_dot[0],
            omega_dot[1],
            omega_dot[2],
        )
        .build();
    let mut mass_matrix = Mat::<f64>::zeros(6, 6);
    mass_matrix
        .diagonal_mut()
        .column_vector_mut()
        .subrows_mut(0, 3)
        .fill(m);
    mass_matrix.submatrix_mut(3, 3, 3, 3).copy_from(&j);
    model.add_mass_element(mass_node_id, mass_matrix);

    let ground_node_id = model.add_node().position_xyz(0., 0., 0.).build();
    model.add_rigid_constraint(mass_node_id, ground_node_id);
    model.add_prescribed_constraint(ground_node_id);

    // Set gravity
    model.set_gravity(gamma[0], gamma[1], gamma[2]);

    //--------------------------------------------------------------------------
    // run simulation
    //--------------------------------------------------------------------------

    // Create output directory
    fs::create_dir_all(out_dir).unwrap();

    // Create solver
    let mut solver = model.create_solver();

    // Create state
    let mut state = model.create_state();

    // Open output file
    let mut file = File::create(format!("{out_dir}/heavy_top.csv")).unwrap();

    // Rotation vector for an
    let mut rv = Col::<f64>::zeros(3);

    // Time step
    for i in 0..n_steps {
        let t = (i as f64) * time_step;

        // Output current position and rotation
        let u = state.u.col(0).subrows(0, 3);
        let q = state.u.col(0).subrows(3, 4);
        quat_as_rotation_vector(q, rv.as_mut());
        file.write_fmt(format_args!(
            "{},{},{},{},{},{},{}\n",
            t, u[0], u[1], u[2], rv[0], rv[1], rv[2]
        ))
        .unwrap();

        if i == 400 {
            assert_matrix_eq!(
                state.u.col(0).as_2d(),
                col![
                    -0.4220299141898183,
                    -0.09451353137427536,
                    -0.04455341442645723,
                    -0.17794086498990777,
                    0.21672292516262048,
                    -0.9597292673920982,
                    -0.016969254156485276
                ]
                .as_2d(),
                comp = float
            );
        }

        // Step
        let res = solver.step(&mut state);

        assert_eq!(res.converged, true);
    }
}

#[test]
#[ignore]
fn test_rigid_platform() {
    let out_dir = "output/rigid_platform";
    let time_step: f64 = 0.01;
    let t_end: f64 = 120.;
    let n_steps = ((t_end / time_step).ceil() as usize) + 1;

    // Construct platform mass matrix
    let gravity = -9.8124;
    let platform_mass = 1.419625E+7;
    let platform_moi = [1.2898E+10, 1.2851E+10, 1.4189E+10];
    let platform_cm_position = col![0., 0., -7.53];
    let platform_mass_matrix = mat![
        [platform_mass, 0., 0., 0., 0., 0.],
        [0., platform_mass, 0., 0., 0., 0.],
        [0., 0., platform_mass, 0., 0., 0.],
        [0., 0., 0., platform_moi[0], 0., 0.],
        [0., 0., 0., 0., platform_moi[1], 0.],
        [0., 0., 0., 0., 0., platform_moi[2]],
    ];

    // Mooring line properties
    let mooring_line_stiffness = 48.9e3; // N
    let mooring_line_length = 55.432; // m

    // Create platform
    let mut platform = RigidPlatform {
        gravity: [0., 0., gravity],
        node_id: 0,
        node_position: col![
            platform_cm_position[0],
            platform_cm_position[1],
            platform_cm_position[2],
            1.,
            0.,
            0.,
            0.
        ],
        mass_matrix: platform_mass_matrix,
        mooring_lines: vec![
            MooringLine {
                stiffness: mooring_line_stiffness,
                unstretched_length: mooring_line_length,
                fairlead_node_position: col![-40.87, 0.0, -14.],
                fairlead_node_id: 0,
                anchor_node_position: col![-105.47, 0.0, -58.4],
                anchor_node_id: 0,
            },
            MooringLine {
                stiffness: mooring_line_stiffness,
                unstretched_length: mooring_line_length,
                fairlead_node_position: col![20.43, -35.39, -14.],
                fairlead_node_id: 0,
                anchor_node_position: col![52.73, -91.34, -58.4],
                anchor_node_id: 0,
            },
            MooringLine {
                stiffness: mooring_line_stiffness,
                unstretched_length: mooring_line_length,
                fairlead_node_position: col![20.43, 35.39, -14.],
                fairlead_node_id: 0,
                anchor_node_position: col![52.73, 91.34, -58.4],
                anchor_node_id: 0,
            },
        ],
    };

    // Model
    let mut model = Model::new();
    model.set_gravity(
        platform.gravity[0],
        platform.gravity[1],
        platform.gravity[2],
    );
    model.set_solver_tolerance(1e-5, 1.);
    model.set_time_step(time_step);
    model.set_rho_inf(0.9);
    model.set_max_iter(6);

    // Add platform node
    platform.node_id = model
        .add_node()
        .position(
            platform.node_position[0],
            platform.node_position[1],
            platform.node_position[2],
            platform.node_position[3],
            platform.node_position[4],
            platform.node_position[5],
            platform.node_position[6],
        )
        .build();

    // Add platform mass element
    model.add_mass_element(platform.node_id, platform.mass_matrix);

    platform.mooring_lines.iter_mut().for_each(|ml| {
        // Add fairlead nodes for mooring attachment
        ml.fairlead_node_id = model
            .add_node()
            .position_xyz(
                ml.fairlead_node_position[0],
                ml.fairlead_node_position[1],
                ml.fairlead_node_position[2],
            )
            .build();

        // Add rigid constraints between platform node and fairlead node
        model.add_rigid_constraint(platform.node_id, ml.fairlead_node_id);

        // Add anchor nodes for mooring attachment
        ml.anchor_node_id = model
            .add_node()
            .position_xyz(
                ml.anchor_node_position[0],
                ml.anchor_node_position[1],
                ml.anchor_node_position[2],
            )
            .build();

        // Add fixed constraint to anchor node
        model.add_prescribed_constraint(ml.anchor_node_id);

        // Add spring element between fairlead node and anchor node
        model.add_spring_element(
            ml.fairlead_node_id,
            ml.anchor_node_id,
            ml.stiffness,
            Some(ml.unstretched_length),
        );
    });

    //--------------------------------------------------------------------------
    // run simulation
    //--------------------------------------------------------------------------

    // Create output directory
    fs::create_dir_all(out_dir).unwrap();

    // Create solver
    let mut solver = model.create_solver();

    // Create state
    let mut state = model.create_state();

    // Get DOFs for applying force
    let platform_z_dof = solver.nfm.get_dof(platform.node_id, Direction::Z).unwrap();
    let platform_rx_dof = solver.nfm.get_dof(platform.node_id, Direction::RX).unwrap();
    let platform_ry_dof = solver.nfm.get_dof(platform.node_id, Direction::RY).unwrap();
    let platform_rz_dof = solver.nfm.get_dof(platform.node_id, Direction::RZ).unwrap();

    // Calculate buoyancy force to balance gravity and mooring lines
    solver.elements.springs.calculate(&state);
    let fm = solver.elements.springs.f.row(2).sum();
    let fp = -gravity * platform_mass;
    let fb = 1.01 * (fm + fp);
    solver.fx[platform_z_dof] = fb;

    // Open output file
    let mut file = File::create(format!("{out_dir}/motion.csv")).unwrap();
    let mut rv = Col::<f64>::zeros(3);

    // Get list of node ID pairs for writing vtk lines
    let vtk_platform_lines = platform
        .mooring_lines
        .iter()
        .map(|ml| [platform.node_id, ml.fairlead_node_id])
        .collect_vec();

    let vtk_mooring_lines = platform
        .mooring_lines
        .iter()
        .map(|ml| [ml.fairlead_node_id, ml.anchor_node_id])
        .collect_vec();

    // let mut fb_rot = Col::<f64>::zeros(3);
    // let mut platform_r_inv = Col::<f64>::zeros(4);

    // Time step
    for i in 0..n_steps {
        let t = (i as f64) * time_step;

        // Output current position and rotation
        state.calculate_x();
        let u = state.x.col(platform.node_id).subrows(0, 3);
        let q = state.x.col(platform.node_id).subrows(3, 4);
        quat_as_rotation_vector(q, rv.as_mut());
        file.write_fmt(format_args!(
            "{},{},{},{},{},{},{}\n",
            t, u[0], u[1], u[2], rv[0], rv[1], rv[2]
        ))
        .unwrap();

        if i % 10 == 0 {
            lines_as_vtk(&vtk_platform_lines, &state)
                .export_ascii(format!("{out_dir}/platform_{:0>5}.vtk", i / 10))
                .unwrap();

            lines_as_vtk(&vtk_mooring_lines, &state)
                .export_ascii(format!("{out_dir}/mooring_{:0>5}.vtk", i / 10))
                .unwrap();
        }

        // Apply moments to platform node
        solver.fx[platform_rx_dof] = 5.0e5 * (2. * PI / 15. * t).sin();
        solver.fx[platform_ry_dof] = 1.0e6 * (2. * PI / 30. * t).sin();
        solver.fx[platform_rz_dof] = 2.0e7 * (2. * PI / 60. * t).sin();

        // Step
        let res = solver.step(&mut state);

        // println!("t={:?}", t);
        // println!("fb={:?}", fb_rot);
        // println!("f={:?}", solver.elements.springs.f);
        // println!("phi={:?}", solver.phi);
        // println!("b={:?}", solver.b);
        // let fl_r_x = Col::<f64>::from_fn(3, |i| {
        //     solver.r[solver
        //         .nfm
        //         .get_dof(platform.mooring_lines[i].fairlead_node_id, Direction::X)
        //         .unwrap()]
        // });
        // let fl_r_z = Col::<f64>::from_fn(3, |i| {
        //     solver.r[solver
        //         .nfm
        //         .get_dof(platform.mooring_lines[i].fairlead_node_id, Direction::Z)
        //         .unwrap()]
        // });
        // println!("fl_r_x={:?}, sum={}", fl_r_x, fl_r_x.sum());
        // println!("fl_r_z={:?}, sum={}", fl_r_z, fl_r_z.sum());
        // println!("l={:?}", solver.elements.springs.l);

        assert_eq!(res.converged, true);
    }
}
