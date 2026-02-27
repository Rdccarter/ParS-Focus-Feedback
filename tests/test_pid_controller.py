from auto_focus.pid_controller import PidConfig, PidController


def test_pid_deadband_no_apply():
    pid = PidController(PidConfig(command_deadband_um=0.1, kp=0.1, ki=0.0, kd=0.0), initial_output=0.0)
    corr, applied = pid.update(0.5, 0.01)
    assert applied is False
    assert corr == 0.0


def test_pid_applies_and_updates_output():
    pid = PidController(PidConfig(kp=1.0, ki=0.0, kd=0.0, max_step_um=1.0, command_deadband_um=0.0), initial_output=0.0)
    corr, applied = pid.update(0.2, 0.1)
    assert applied is True
    assert corr < 0
    assert pid.output == corr
