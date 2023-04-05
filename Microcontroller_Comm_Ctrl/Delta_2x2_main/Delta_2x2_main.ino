#include "delta_trajectory.pb.h"
#include "pb_common.h"
#include "pb.h"
#include "pb_encode.h"
#include "pb_decode.h"
#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include <Adafruit_ADS1X15.h>
#include <math.h>
#include "myServoController.h"

#define NUM_MOTORS 12 
#define MY_ID 10

//######################################## Timer ###################
#define CPU_HZ 48000000
#define TIMER_PRESCALER_DIV 1024

//void startTimer(int frequencyHz);
//void setTimerFrequency(int frequencyHz);
//void TC3_Handler();


//################################## Feather MC and ADC Libraries INIT #####################3
Adafruit_MotorShield MC0 = Adafruit_MotorShield(0x62);
Adafruit_MotorShield MC1 = Adafruit_MotorShield(0x60);
Adafruit_MotorShield MC2 = Adafruit_MotorShield(0x61);
//
Adafruit_DCMotor *MC0_M1 = MC0.getMotor(1);
Adafruit_DCMotor *MC0_M2 = MC0.getMotor(2);
Adafruit_DCMotor *MC0_M3 = MC0.getMotor(3);
Adafruit_DCMotor *MC0_M4 = MC0.getMotor(4);
Adafruit_DCMotor *MC1_M1 = MC1.getMotor(1);
Adafruit_DCMotor *MC1_M2 = MC1.getMotor(2);
Adafruit_DCMotor *MC1_M3 = MC1.getMotor(3);
Adafruit_DCMotor *MC1_M4 = MC1.getMotor(4);
Adafruit_DCMotor *MC2_M1 = MC2.getMotor(1);
Adafruit_DCMotor *MC2_M2 = MC2.getMotor(2);
Adafruit_DCMotor *MC2_M3 = MC2.getMotor(3);
Adafruit_DCMotor *MC2_M4 = MC2.getMotor(4);
//
Adafruit_DCMotor* motors[NUM_MOTORS] = {MC0_M1,MC0_M2,MC1_M1,// 1st robot
                                        MC1_M2,MC2_M1,MC2_M2,// 2nd robot
                                        MC2_M3,MC2_M4,MC1_M3,// 3rd robot
                                        MC1_M4,MC0_M3,MC0_M4,// 4th robot
                                        };
//Adafruit_DCMotor* motors[NUM_MOTORS] = {MC1_M4};


Adafruit_ADS1015 ADC2;
Adafruit_ADS1015 ADC1;
Adafruit_ADS1015 ADC0;

Adafruit_ADS1015* adcs[NUM_MOTORS] = {&ADC2, &ADC2, &ADC1,//1st robot
                                      &ADC1, &ADC0, &ADC0,//2nd robot
                                      &ADC0, &ADC0, &ADC1,//3rd robot
                                      &ADC1, &ADC2, &ADC2,//4th robot
                                      };

int channels[NUM_MOTORS] = {0,1,0,//1st robot
                            1,0,1,//2nd robot
                            2,3,2,//3rd robot
                            3,2,3,//4th robot
                            };

//#################################### GLOBAL VARIABLES ##################################//
const int numChars = 3000;
float pi = 3.1415926535;

uint8_t input_cmd[numChars];
bool newData = false;

DeltaMessage message = DeltaMessage_init_zero;
static bool recvInProgress = false;
int ndx = 0;
int ndxx = 0;
char startMarker = 0xA6;
char confMarker = '~';
char endMarker = 0xA7;
int marker_count = 0;
bool skip_trajectory = false;

float trajectory[20][NUM_MOTORS];
int traj_iter = 0;
bool go = false;
bool reset_flag = false;
bool is_movement_done = false;


myServoController controller(motors, adcs, channels);


void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);
  while (!Serial1) {
    delay(10);
  }
  // set all the base dc motor control pins to outputs
  MC0.begin();
  MC1.begin();
  MC2.begin();
  // start all the ADCs
  ADC2.begin(0x4A);
  ADC1.begin(0x49);
  ADC0.begin(0x48);

  ADC2.setGain(GAIN_ONE);
  ADC1.setGain(GAIN_ONE);
  ADC0.setGain(GAIN_ONE);
  
  controller.init_controller();
  init_comms();

  //startTimer(20); //20 Hz -> 50ms update freq
  
  Serial.print("q"); Serial.print('\t');
  Serial.print("qCmd"); Serial.print('\t');
  Serial.print("q_dot"); Serial.print('\t');
  Serial.println("q_dotCmd");
  
}

float xmin = 0.005;
float xmax = 0.095;

long m_prev = millis();
void loop() {
  
  recvWithStartEndMarkers();
  if (newData) {
    if (decodeNanopbData() & !skip_trajectory) {
        updateTrajectory();
    }
    newData = false;
    ndx = 0;
  }
  executeWaypoints();
  if (traj_iter >= 20 && !reset_flag) {
//    reset();
//    reset_flag = true;
  }
}
