echo "Welcome to DriveSafe"

echo "Do you want to enter the car? [Y/N]"
read input

if [ $input == 'y' ]
then
    echo "Scan face for recognition"
    `python ./src/facial_recognition.py`
    echo "Face recognized"
    echo "Welcome!"
    echo "Car unlocked"
    echo "You are now connected to the car"
    `python ./src/behaviour_detection.py`
    echo "Thank you using DriveSafe."
    echo "Good Bye"
else
    echo "Goodbye"
fi
