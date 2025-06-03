//Author: Christian Shamo
//Date: 06/21/2024
//Description: This code is made for the Rodney V4 PCB. It tells multiplexor to switch between 4 channels (pins) to get readings from 4 different adcs. The adcs each have two readings,
//one on analog pin 0 and one on analog pin 1. Each adc is responsible for one of the 4 wheatstone half bridges on Rodney. The difference between analog pin 0 and analog pin 1 on each 
//adc is the voltage difference for the wheatstone half bridge that we care about, in its raw analog form. The code takes in the continous differential (the difference is programmed to
//be already calculated using the adc library) readings from all 4 adcs as fast as it can and prints them to the serial port. A corresponding python file reads the data in from the 
//serial port. This constantly looks to see if a push-button is pressed. When the button is pressed the first time, the python program is programmed to begin storing the incoming data
//from the serial port into a .csv file. When the push-button is pressed a second time, the python program stops storing incoming data. This Arduino application does not have to be open
//for everything to work - after this code has been flashed to the Arduino Uno. Only the python program needs to be open. 

#include <Wire.h>              //library for controlling I2C communication and controlling the mux (is not mux specific)
#include <Adafruit_ADS1X15.h>  //library for programming the ADS1115 ADCs

#define PCA9546A_ADDR 0x70     //Mux's address

#define start_signal '#'
#define reset_signal '$'

int button = 2;
bool py_send = false;
unsigned long time_init;
int16_t diff0, diff1, diff2, diff3;

Adafruit_ADS1115 ads1;
Adafruit_ADS1115 ads2;
Adafruit_ADS1115 ads3;
Adafruit_ADS1115 ads4;

void pca9546a_select(uint8_t bus) {
  if (bus > 3) return;
  Wire.beginTransmission(PCA9546A_ADDR);
  Wire.write(1 << bus);
  Wire.endTransmission();
}

void initializeADS() {
  // Initialize each ADS1115 on its corresponding bus
  pca9546a_select(0);
  if (!ads1.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 0");
    while (1);
  }

  // Set the data rate to the ads1115's max speed option of 860 samples per second.
  ads1.setDataRate(RATE_ADS1115_860SPS);
  // Set the amplifier gain of the raw data
  ads1.setGain(GAIN_SIXTEEN);
  // Set the adc in continous sampling mode (otherwise it'll be much slower). Tell it to take the difference between analog pins 0 and 1 on the adc.
  ads1.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);
  
  pca9546a_select(1);
  if (!ads2.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 1");
    while (1);
  }
  ads2.setDataRate(RATE_ADS1115_860SPS);
  ads2.setGain(GAIN_SIXTEEN);
  ads2.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);
  
  pca9546a_select(2);
  if (!ads3.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 2");
    while (1);
  }
  ads3.setDataRate(RATE_ADS1115_860SPS);
  ads3.setGain(GAIN_SIXTEEN);
  ads3.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);
  
  pca9546a_select(3);
  if (!ads4.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 3");
    while (1);
  }
  ads4.setDataRate(RATE_ADS1115_860SPS);
  ads4.setGain(GAIN_SIXTEEN);
  ads4.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);
}

void setup() {
  Serial.begin(115200);
  // Sets the I2C bus speed. Fast mode is 400 kHz. Standard mode is just 100 kHz. Super fast mode is faster than 400 kHz, but the ADCs are limited to 400 kHz.
  Wire.setClock(400000);
  Wire.begin(); // Initialize I2C communication
  
  pinMode(button, INPUT_PULLUP);

  initializeADS();
  
}

void loop() {
  
  // Read differential from ADS1115 on bus 0. I've tried a million ways to get the data as fast as possible without compromising accuracy. This seems to be the best way. 
  pca9546a_select(0);
  diff0 = ads1.getLastConversionResults();

  // Read differential from ADS1115 on bus 1
  pca9546a_select(1);
  diff1 = ads2.getLastConversionResults();
 
  // Read differential from ADS1115 on bus 2
  pca9546a_select(2);
  diff2 = ads3.getLastConversionResults();

  // Read differential from ADS1115 on bus 3
  pca9546a_select(3);
  diff3 = ads4.getLastConversionResults();

  
  // Check if the button is pressed and send the start signal (only used in the python file to start keeping track of data)
  if (digitalRead(button) == LOW and py_send == false) {
    Serial.println(start_signal); // Send the start signal
    py_send = true;
    delay(750);
    time_init = micros();
  }
 
  // Check if the button is pressed again to send the end test signal (only used in the python file to start keeping track of data)
  if (digitalRead(button) == LOW and py_send == true) {
    Serial.println("test ended");
    py_send = false;
  }

  if (diff0 == 0 and diff1 == 0 and diff2 == 0 and diff3 == 0) {
    Wire.end();
    delay(10);
    Wire.begin();
    initializeADS();
    Serial.println(reset_signal); // Send the reset signal
    Serial.print(",");
    Serial.print(String(micros() - time_init)); 
    return;
  }
  
  // I found this to be the most efficient way (sampling rate-wise) to print the data to the serial port
  Serial.print(String(micros() - time_init)); 
  Serial.print(","); 
  Serial.print(String(diff0)); 
  Serial.print(","); 
  Serial.print(String(diff2)); 
  Serial.print(","); 
  Serial.print(String(diff1)); 
  Serial.print(","); 
  Serial.println(String(diff3));
  
}
