# First "lsusb"  command on terminal to get the usb port of the scope 
# sudo chmod a+rw /dev/bus/usb/001/006  

import pyvisa
rm = pyvisa.ResourceManager('@py')
resources = rm.list_resources()
usb_resources = [r for r in resources if "USB" in r.upper()]

print(usb_resources)