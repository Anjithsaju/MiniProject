import pyttsx3
def speach(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    print("hello")
with open('new.txt', 'r') as file:
    
    lines = file.readlines()

    
    for line in lines:
     
        line = line.strip()
        
       
        content = line.split('_')
        
      
        rupees = content[0]
        
       
        speach(rupees+"rupees")





