#include <SPI.h>
#include <DMD.h>        
#include <TimerOne.h>   
#include "SystemFont5x7.h"
#include "Arial_black_16.h"
#define ROW_MODULE 1
#define COLUMN_MODULE 1

DMD p10(ROW_MODULE, COLUMN_MODULE);

char message[100];
char char_read;
byte pos_index = 0;
int i;            
char welcome_screen[] = "Welcome";

void p10scan()
{ 
  p10.scanDisplayBySPI();
}

void setup()
{
   Timer1.initialize(1000);
   Timer1.attachInterrupt(p10scan);
   p10.clearScreen( true );
   Serial.begin(9600);
   strcpy(message,welcome_screen);
}
void loop()
{
   if(Serial.available())
   {           
        for(i=0; i<99; i++)
        {
          // Set each element in array back to 0 (null value)
            message[i] = '\0';
        }      
        pos_index=0;
    }

// Loop through array of income stream
    while(Serial.available() > 0)
    {
       if(pos_index < (99)) 
       {   
          // replace each element / char with message char      
           char_read = Serial.read();
           message[pos_index] = char_read;
           pos_index++;      
       } 
   }
   // Select font type 
   p10.selectFont(Arial_Black_16);
   // Draw text on right side
   p10.drawMarquee(message ,100,(32*ROW_MODULE)-1,0);

   // Scrolling
   long start=millis();
   long timer_start=start;
   boolean flag=false;
   while(!flag)
   {
     // control speed by changing '15'
     if ((timer_start+15) < millis()) 
     {
       flag=p10.stepMarquee(-1,0);
       timer_start=millis();
     }
   }
}
