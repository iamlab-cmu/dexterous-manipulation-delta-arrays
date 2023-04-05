#ifndef myServoController_h
#define myServoController_h
#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include <Adafruit_ADS1X15.h>
#include <math.h>

#define NUM_MOTORS 12


class myServoController {
public:
  myServoController(Adafruit_DCMotor* motors[NUM_MOTORS], 
                    Adafruit_ADS1015* adcs[NUM_MOTORS], 
                    int channels[NUM_MOTORS]);
  void init_controller();
  void reset_joints();
  void stop_moving();
  void cmdPos(float qCmd[]);
  void cmdVel(float q_dotCmd[]);
  void driveController();
  void print_state();

  // State estimation
  float _joint_positions[NUM_MOTORS];
  float _prev_joint_positions[NUM_MOTORS];
  float _joint_velocities[NUM_MOTORS];

  // Control
  float _joint_positionsCmd[NUM_MOTORS];
  float _joint_velocitiesCmd[NUM_MOTORS];
  int _motorCmd[NUM_MOTORS];

  // trajectory
  int _refCmd[NUM_MOTORS];
  float _traj_t0;
  float _traj_tf;
  float _traj_t;
  void ramp2pos(float xf[NUM_MOTORS], float vmax, float amax);
  
private:
  // Position
  Adafruit_ADS1015* _adcs[NUM_MOTORS];
  int _channels[NUM_MOTORS];
  void read_joint_positions();

  // Actuator
  Adafruit_DCMotor *_motors[NUM_MOTORS];

  long _millis;
  long _delta_millis;

  // cmdPos
  void positional_PID(float PID_Cmd[NUM_MOTORS]);
  // Note these constants were determined with error in meters
  const float _cmdPos_Kp = 100000;
  const float _cmdPos_Ki = 1000;
  const float _cmdPos_Kd = 400;
  float _cmdPos_error[NUM_MOTORS];
  float _cmdPos_integral_error[NUM_MOTORS];
  float _cmdPos_derivative_error[NUM_MOTORS];


  // driveController
  void update_state();
  void send_motorCmd();
  const float _lambda_cmdPos = 0.5;

  // trajectory
  void ramp_pos(float t, float qCmd[NUM_MOTORS], float x0[NUM_MOTORS], float xf[NUM_MOTORS], float v[NUM_MOTORS], float amax_ref);
  void ramp_vel(float t, float q_dotCmd[NUM_MOTORS], float x0[NUM_MOTORS], float xf[NUM_MOTORS], float v[NUM_MOTORS], float amax_ref);
  float ramp_tf(float tf[NUM_MOTORS], float x0[NUM_MOTORS], float xf[NUM_MOTORS], float vmax, float amax);

  
};
#endif
