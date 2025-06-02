# kandi

## main.cpp: 
Yksi useasta ESP32 toteutuksesta. Tähän on implementoitu ESP32 testit. Se on kuuden vapausasteen floating point kinematiikkaketju joka kuvastaa Piperia.

## main.py: 
Projektin alkuperäinen prototyyppi. Visualisoi klamptia käyttäen pythonilla ajettua kinematiikkaketjun mallinnusta. Kehitetty kaavojen tarkistamista varten projektin alkuvaiheessa.

## pipermanual.urdf:
lähde: https://github.com/agilexrobotics/piper_ros/tree/noetic/src/piper_description/urdf
Tästä muokattu versio jota väritetty klamptia varten jotta nivelet pystyy hahmottamaan

## drawlines.py ja ulkoinenjasisainen.py:
Käytetty hahmottamisen apuna (lähinnä kuvia varten). Käyttää pyplottia.

## keyboardandendeffectorposition
Näppäimiä QWERTY ja 123456 painamalla pystyy liikuttamaan Piper robottia Klamp ympäristössä. Tulostaa päätekappaleen sijaintia terminaaliin.

## hcvalues.py
Muuntaa valitun URDF tiedoston sarjaksi homogeenisiä muunnosmatriiseja jotka kuvastavat staattista osaa, robotin nolla-asentoa.
Aja terminaalissa komennolla: "python hcvalues.py > hcvalues.h"
