from mqtt_wrapper import MQTTWrapper
import distractions
import json
import sys
import time

def loadUserConfig():
    with open(sys.path[0] + "/userConfig.json") as json_file:
        return json.load(json_file)

client = MQTTWrapper.fromJsonConfig(sys.path[0] + "/mqttConfig.json")
user = loadUserConfig()

distractions.webcamSetup()

while True:
    try:
        distractionMap = distractions.getDistraction()
        dataToTransmit = { 
            user["user"]: {
                "drinking": distractionMap["Drinking"],
                "brushing_hair": distractionMap["Brushing hair"],
                "safe_driving": distractionMap["Safe driving"],
                "talking_phone": distractionMap["Talking phone"],
                "texting_phone": distractionMap["Texting phone"]
            } 
        }
        client.publish(json.dumps(dataToTransmit))
        print(dataToTransmit)
        time.sleep(1)
    except distractions.NoDriverDetectedException:
        print("No driver found!")

    if cv2.waitKey(0) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()