/*
Ultrasonic sensor for Jetbot
----------------------------
Trigger on port 11
Respons on port 12
5V to 3.3V response conversion with 2.2K and 4.7K resistors
I2C slave sender on port 8
----------------------------
This sketch is based on the Ping))) Sensor sketch created by David A. Mellis on 3 Nov 2008, modified by Tom Igoe on 30 Aug 2011 and modified by Ron in 2020 to 
include I2C communications.
----------------------------
The original source of Ping))) can be found on:
  http://www.arduino.cc/en/Tutorial/Ping
This example code is in the public domain.
*/

#include <Wire.h>

// this constant won't change. 
// It's the pin number of the sensor's output:
const int pingPin = 11; // TRIG
const int distPin = 12; // ECHO
long cm;

void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  pinMode(pingPin, OUTPUT);
  pinMode(distPin, INPUT);
  Wire.begin(8);
  Wire.onRequest(requestEvent);
}

void loop() {
  // establish variables for duration of the ping, 
  long duration;
  
  // The PING))) is triggered by a HIGH pulse of 2 
  // or more microseconds.
  // Give a short LOW pulse beforehand to ensure a clean 
  // HIGH pulse:
  digitalWrite(pingPin, LOW);
  delayMicroseconds(2);
  digitalWrite(pingPin, HIGH);
  delayMicroseconds(5);
  digitalWrite(pingPin, LOW);
  
  // A HIGH pulse whose duration is the time (in 
  // microseconds) from the sending of the ping
  // to the reception of its echo off of an object.
  duration = pulseIn(distPin, HIGH);
  
  // convert the time into a distance
  cm = microsecondsToCentimeters(duration);
  // Print distance to serial monitor for testing
  Serial.print(cm);
  Serial.print("cm");
  Serial.println();
  delay(100);
}

// I2C function that executes whenever data is requested 
// by master. This function is registered as an event, 
// see setup()
void requestEvent() {
  byte buf[4];
  buf[0] = (byte) cm;
  buf[1] = (byte) cm>>8;
  buf[2] = (byte) cm>>16;
  buf[3] = (byte) cm>>24;
  Wire.write(buf, 4); // respond with message of 4 bytes 
                      // (cm is long type)
  Serial.println("I2C requested");
}


long microsecondsToCentimeters(long microseconds) {
  // The speed of sound is 340 m/s or 29 microseconds 
  // per centimeter.
  // The ping travels out and back, so to find the 
  // distance of the object we take half of the distance 
  // travelled.
  return microseconds / 29 / 2;
}
