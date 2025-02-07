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
                "closed_eyes": distractionMap["Closed eyes"],
                "safe_driving": distractionMap["Safe driving"],
                "smoking": distractionMap["Smoking"],
                "talking_phone": distractionMap["Talking phone"],
                "texting_phone": distractionMap["Texting phone"],
                "turning": distractionMap["Turning"]
            } 
        }
        client.publish(json.dumps(dataToTransmit))
        print(dataToTransmit)
        time.sleep(1)
    except NoFaceDetectedException:
        print("No face found!")

    if cv2.waitKey(0) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()