path = "/home/pi/Downloads/sensor_Data.csv"
import spidev
from time import sleep
import os
import glob
import time 
from datetime import datetime


## To open the SPI bus

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz =7629          ## max driver freq

os.system('modprobe w1-gpio')        ## Enabling GPIO PIN

os.system('modprobe w1-therm')       ## Enabling the temp sensor
 
base_dir = '/sys/bus/w1/devices/'               ## address location of temperature sensor
device_folder = glob.glob(base_dir + '28*')[0]  ## retriving the address
device_file = device_folder + '/w1_slave'       ## to display raw data



## Different channels used by MCP3208 Analog To Digital Converter

xchannel = 0
ychannel = 1
zchannel = 2
vchannel = 3
j=0



def getReading(channel):


## SPI Communication with MCP3208  
    
    ## pulling the raw data from the chip
    rawData = spi.xfer([6, (0+channel) << 6, 0])       ###  12 BIT CHANNEL
    #rawData = spi.xfer([4 | 2 | (channel >> 2), (channel & 3) << 6, 0])    --- OTHER WAY TO WRITE THE ABOVE STATEMENT

    ## Rawdata to Bit value
    processedData = ((rawData[1]&15) << 8) + rawData[2]
    return processedData        

def convertVoltage(bitValue, decimalPlaces=2):
    voltage = (bitValue * 3.3) / float(4095)
    voltage = round(voltage, decimalPlaces)
    return voltage

def convertGx(voltage, decimalPlaces=2):
    gvaluex = (voltage - 1.65)/float(0.33)
    gvaluex = round(gvaluex, decimalPlaces)
    return gvaluex

def convertGy(voltage, decimalPlaces=2):
    gvaluey = (voltage - 1.64)/float(0.33)
    gvaluey = round(gvaluey, decimalPlaces)
    return gvaluey

def convertGz(voltage, decimalPlaces=2):
    gvaluez = (voltage - 1.700)/float(0.325)
    gvaluez = round(gvaluez, decimalPlaces)
    return gvaluez

def read_temp_raw():
    f = open(device_file, 'r')
    lines = f.readlines()
    f.close()
    return lines
 
def read_temp():
    lines = read_temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    equals_pos = lines[1].find('t=')
    if equals_pos != -1:
        temp_string = lines[1][equals_pos+2:]
        temp_c = float(temp_string) / 1000.0
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        return temp_c, temp_f


while j<3141:                

## reading the data from accelerometer adxl335
    
    xData = getReading(xchannel)
    yData = getReading(ychannel)
    zData = getReading(zchannel)

## reading the data from vibration sensor ADIS16220
    
    vData = getReading(vchannel)


    xVoltage = convertVoltage(xData)
    yVoltage = convertVoltage(yData)
    zVoltage = convertVoltage(zData)

## conversion of acceleration data to g's
    
    xG = convertGx(xVoltage)
    yG = convertGy(yVoltage)
    zG = convertGz(zVoltage)

## reading the data from temperature sensor DS18B20
    
    temp_c,temp_f = read_temp()



## outputting the sensor data from all the sensors
    
    print("x bitValue = {} ; Voltage = {} V ; xGvalue = {} g".format(xData, xVoltage, xG))
    print("y bitValue = {} ; Voltage = {} V ; yGvalue = {} g".format(yData, yVoltage, yG))
    print("z bitValue = {} ; Voltage = {} V ; zGvalue = {} g".format(zData, zVoltage, zG))
    print("v bitValue = {} ".format(vData))
    print("temp_c = {} ".format(temp_c))
    print("temp_f = {} ".format(temp_f))

    sleep(sleepTime)



## writing the retrived data to a csv file which will be created in the path given in the first line of this code

    file = open(path, "a")
    if os.stat(path).st_size == 0:
            file.write("X1,Y1,Z1\n")
    file.write(str(xG)+","+str(yG)+","+str(zG)+"\n")
    file.flush()
    j=j+1

file.close()
