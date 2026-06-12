use diffsol::{
    BdfState, NalgebraContext, NalgebraLU, NalgebraMat, NalgebraVec, OdeBuilder, OdeSolverMethod,
    OdeSolverStopReason, VectorHost,
};

type M = NalgebraMat<f64>;
type LS = NalgebraLU<f64>;

const K: f64 = 0.1;
const T_END: f64 = 10.0;
const N_TARGETS: usize = 100;
const H0: f64 = 1e-3;
const RTOL: f64 = 1e-6;
const ATOL: f64 = 1e-8;

fn make_problem(
    t0: f64,
    y0: f64,
) -> diffsol::OdeSolverProblem<
    impl diffsol::OdeEquationsImplicit<M = M, V = NalgebraVec<f64>, T = f64, C = NalgebraContext>,
> {
    OdeBuilder::<M>::new()
        .t0(t0)
        .h0(H0)
        .rtol(RTOL)
        .atol([ATOL])
        .p([K])
        .rhs_implicit(
            |x: &NalgebraVec<f64>, p: &NalgebraVec<f64>, _t: f64, y: &mut NalgebraVec<f64>| {
                y.as_mut_slice()[0] = -p.as_slice()[0] * x.as_slice()[0];
            },
            |_: &NalgebraVec<f64>,
             p: &NalgebraVec<f64>,
             _t: f64,
             v: &NalgebraVec<f64>,
             jv: &mut NalgebraVec<f64>| {
                jv.as_mut_slice()[0] = -p.as_slice()[0] * v.as_slice()[0];
            },
        )
        .init(
            move |_p: &NalgebraVec<f64>, _t: f64, y: &mut NalgebraVec<f64>| {
                y.as_mut_slice()[0] = y0;
            },
            1,
        )
        .build()
        .unwrap()
}

fn target(i: usize) -> f64 {
    i as f64 * T_END / N_TARGETS as f64
}

fn single_sweep() -> usize {
    let problem = make_problem(0.0, 100.0);
    let mut solver = problem.bdf::<LS>().unwrap();
    while solver.state().t < T_END {
        solver.step().unwrap();
    }
    let _ = solver.interpolate(T_END).unwrap();
    solver.get_statistics().number_of_steps
}

fn cold_restart_each_target() -> usize {
    let mut steps = 0;
    let mut t0 = 0.0;
    let mut y0 = 100.0;
    for i in 1..=N_TARGETS {
        let t = target(i);
        let problem = make_problem(t0, y0);
        let mut solver = problem.bdf::<LS>().unwrap();
        solver.set_stop_time(t).unwrap();
        loop {
            if solver.step().unwrap() == OdeSolverStopReason::TstopReached {
                break;
            }
        }
        steps += solver.get_statistics().number_of_steps;
        t0 = t;
        y0 = solver.state().y.as_slice()[0];
    }
    steps
}

fn checkpoint_restart_with_tstop() -> usize {
    let mut steps = 0;
    let mut t0 = 0.0;
    let mut y0 = 100.0;
    let mut checkpoint: Option<BdfState<NalgebraVec<f64>>> = None;
    for i in 1..=N_TARGETS {
        let t = target(i);
        let problem = make_problem(t0, y0);
        let mut solver = if let Some(state) = checkpoint.take() {
            problem.bdf_solver::<LS>(state).unwrap()
        } else {
            problem.bdf::<LS>().unwrap()
        };
        solver.set_stop_time(t).unwrap();
        loop {
            if solver.step().unwrap() == OdeSolverStopReason::TstopReached {
                break;
            }
        }
        steps += solver.get_statistics().number_of_steps;
        t0 = t;
        y0 = solver.state().y.as_slice()[0];
        checkpoint = Some(solver.checkpoint());
    }
    steps
}

fn checkpoint_restart_with_interpolation() -> usize {
    let mut steps = 0;
    let mut t0 = 0.0;
    let mut y0 = 100.0;
    let mut checkpoint: Option<BdfState<NalgebraVec<f64>>> = None;
    for i in 1..=N_TARGETS {
        let t = target(i);
        let problem = make_problem(t0, y0);
        let mut solver = if let Some(state) = checkpoint.take() {
            problem.bdf_solver::<LS>(state).unwrap()
        } else {
            problem.bdf::<LS>().unwrap()
        };
        while solver.state().t < t {
            solver.step().unwrap();
        }
        let y = solver.interpolate(t).unwrap();
        steps += solver.get_statistics().number_of_steps;
        t0 = t;
        y0 = y.as_slice()[0];
        checkpoint = Some(solver.checkpoint());
    }
    steps
}

fn main() {
    println!("BDF checkpoint/restart reproducer");
    println!("ODE: dy/dt = -0.1 y, t in [0, 10], 100 output targets");
    println!("rtol={RTOL}, atol={ATOL}, h0={H0}");
    println!();
    println!("{:<48}{}", "single sweep", single_sweep());
    println!(
        "{:<48}{}",
        "cold restart at each target",
        cold_restart_each_target()
    );
    println!(
        "{:<48}{}",
        "checkpoint/restart + set_stop_time",
        checkpoint_restart_with_tstop()
    );
    println!(
        "{:<48}{}",
        "checkpoint/restart + overshoot/interpolate",
        checkpoint_restart_with_interpolation()
    );
}
