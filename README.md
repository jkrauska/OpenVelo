Gist:

Let's try to make something like a pocket radar using cheap parts.

```text
1. Pi  - $30-$50
Zero 2 W, MicroSD Card, OTG Adapter

2. USB Audio Dongle - $10
✔ USB Sound Card With Mic Input
	•	Movo USB-AC 3.5 TRS

3. Power-Line Noise Filter - $5
(You need this to keep Pi 5V switching noise out of radar IF path.)
✔ LC Filter Module (5–36V, 3A)
	•	“LC Low Pass Filter”
	•	“DC Ripple Filter”
	•	“Interference Suppressor”

4. Preamp - $5
✔ MAX9814 Microphone Amplifier Module
	•	Automatic gain control (AGC)
	•	Selectable gain: 40 dB / 50 dB / 60 dB
	•	Input = electret mic bias style, requires AC coupling
	•	Output = clean amplified audio to sound card

5. Coupling Capacitor - $1
✔ 1 µF X7R Ceramic Capacitor (25V or higher)
Used between CDM324 IF → MAX9814 MIC IN

6. Radar Module $10
✔ CDM324 Doppler Radar Sensor
	•	3.3–5V version
	•	Has 3 pins: VCC, GND, IF

7. Cabling
✔ 3.5mm TRS Male → Bare-Wire “Pigtail” Cable - $5
✔ Breadboard, jumpers
```





```text
                  +-------------------------------------------+
                  |              Raspberry Pi Zero            |
                  |-------------------------------------------|
                  |                                           |
             5V --+-----------------------------+             |
            GND --+------------+                |             |
                  |            |                |             |
                  |            |                |             |
                  |        (via USB OTG)        |             |
                  |              |              |             |
                  |              v              |             |
                  |      +------------------+   |             |
                  |      | USB Audio Card   |   |             |
                  |      | (Movo / CM108)   |   |             |
                  |      |                  |   |             |
                  |      |   MIC IN  <------+---+-- MAX9814 OUT
                  |      |   GND     <------+------ Pi GND
                  |      +------------------+   |
                  +-------------------------------------------+

──────────────────────────────────────────────────────────────────
        CLEAN, FILTERED 5V SUPPLY (very important)
──────────────────────────────────────────────────────────────────

Pi 5V ----[ LC FILTER MODULE ]-------------------- Clean 5V output
                   | 
                   |  (Module has IN+, IN– → OUT+, OUT–)
                   |
                   +----> Clean 5V Rail for Radar + Preamp


──────────────────────────────────────────────────────────────────
                 CDM324 RADAR MODULE WIRING
──────────────────────────────────────────────────────────────────

                +------------------+
 Clean 5V  ---> |  VCC             |
 Pi GND    ---> |  GND             |
 CDM324 IF ---> |  IF   ---||----> (to MAX9814 MIC IN)
                +------------------+       1 µF X7R (AC coupling)


──────────────────────────────────────────────────────────────────
                  MAX9814 PREAMP WIRING
──────────────────────────────────────────────────────────────────

                 +-------------------------------+
 Clean 5V  ----> | VDD                           |
 Pi GND     ----> | GND                           |
 CDM324 IF (via 1µF) --> MIC IN                   |
 MAX9814 OUT -----> To USB Sound Card MIC IN      |
 GAIN pin --------> Tie to GND for **50 dB gain** |
 AR pin ----------> Leave unconnected (default)   |
                 +-------------------------------+


──────────────────────────────────────────────────────────────────
                       GROUNDING NOTES
──────────────────────────────────────────────────────────────────

ALL GROUNDS MUST CONNECT TOGETHER:

- Pi GND
- MAX9814 GND
- CDM324 GND
- USB Sound Card GND (this comes via USB)
- LC Filter output ground

Use a **star ground** if possible:
     one node where all grounds meet → minimizes noise / hum.


──────────────────────────────────────────────────────────────────
```


