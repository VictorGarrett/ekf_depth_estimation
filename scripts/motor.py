import rospy
from std_msgs.msg import Float32

class DCMotor:
    def __init__(self, gain=1.0, time_constant=0.5, rate_hz=100,
                 voltage_topic="/motor_voltage", velocity_topic="/your_joint_velocity_controller/command"):

        # Motor model parameters
        self.K = gain
        self.tau = time_constant
        self.velocity = 0.0
        self.input_voltage = 0.0
        self.last_time = None

        self.voltage_sub = rospy.Subscriber(voltage_topic, Float32, self._voltage_callback)
        self.velocity_pub = rospy.Publisher(velocity_topic, Float32, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self._update)

        self._wait_for_clock()

    def _wait_for_clock(self):
        while rospy.Time.now().to_sec() == 0 and not rospy.is_shutdown():
            rospy.loginfo_throttle(2, "Waiting for simulation time (from /clock)...")
            rospy.sleep(0.1)

    def _voltage_callback(self, msg):
        self.input_voltage = msg.data

    def _update(self, event):
        current_time = rospy.get_time()
        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        self.last_time = current_time

        dv = (self.K * self.input_voltage - self.velocity) / self.tau
        self.velocity += dv * dt

        self.velocity_pub.publish(self.velocity)

    def get_velocity(self):
        return self.velocity