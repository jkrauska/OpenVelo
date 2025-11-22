Gist:

Let's try to make something like a pocket radar using cheap parts.




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


```mermaid
flowchart LR

  %% Power path
  PSU[5V Power Supply]
  PI5V[Pi 5V Pin]
  PIGND[Pi GND Pin]

  PSU --> PI5V
  PSU --> PIGND

  PI5V --> LC_IN[LC Filter Input Plus]
  PIGND --> LC_IN_GND[LC Filter Input Ground]

  LC_IN --> LC_OUT[LC Filter Output Clean 5V]
  LC_IN_GND --> LC_OUT_GND[LC Filter Output Ground]

  LC_OUT --> RAD_VCC[CDM324 VCC]
  LC_OUT_GND --> RAD_GND[CDM324 GND]

  LC_OUT --> PRE_VDD[MAX9814 VDD]
  LC_OUT_GND --> PRE_GND[MAX9814 GND]

  %% Signal path
  RAD_IF[CDM324 IF] --> CAP[One Microfarad Coupling Capacitor]
  CAP --> PRE_IN[MAX9814 Mic Input]

  PRE_OUT[MAX9814 Output] --> TRS_CABLE[TRS Pigtail Cable]
  TRS_CABLE --> USB_MIC_IN[USB Sound Card Mic Input]

  USB_CARD[USB Sound Card] --> OTG_ADAPTER[USB OTG Adapter]
  OTG_ADAPTER --> PI_USB[Pi Zero USB Port]

  %% Grounding
  RAD_GND --- STAR_GND[Common Ground Point]
  PRE_GND --- STAR_GND
  LC_OUT_GND --- STAR_GND
  PIGND --- STAR_GND

  %% Notes
  GAIN_NOTE:::note
  AR_NOTE:::note

  PRE_IN -.-> GAIN_NOTE
  PRE_IN -.-> AR_NOTE

  classDef note fill:#eef,stroke:#88a,stroke-width:1px,color:#000;
  GAIN_NOTE[MAX9814 Gain Pin Connect to Ground for 50 dB]
  AR_NOTE[MAX9814 AR Pin Leave Unconnected]
```

Pin Out
```mermaid
flowchart TB

  subgraph PI[Pi Zero Header Pins]
    PI_PIN2[Pin 2 5V]
    PI_PIN6[Pin 6 GND]
    PI_USBPORT[Pi USB Data Port]
  end

  subgraph LC[LC Filter Module Pins]
    LC_INP[IN Plus]
    LC_ING[IN Ground]
    LC_OUTP[OUT Clean 5V]
    LC_OUTG[OUT Ground]
  end

  subgraph RAD[CDM324 Pins]
    RAD_VCC[VCC]
    RAD_IF[IF]
    RAD_GND[GND]
  end

  subgraph PRE[MAX9814 Pins]
    PRE_VDD[VDD]
    PRE_GND[GND]
    PRE_IN[Mic Input]
    PRE_OUT[Out]
    PRE_GAIN[Gain]
    PRE_AR[AR]
  end

  subgraph USB[USB Sound Card]
    USB_MIC[Mic Input Jack]
    USB_GND[Mic Ground]
    USB_USB[USB Plug]
  end

  CAP[One Microfarad Coupling Capacitor]
  TRS[TRS Pigtail Cable]

  %% Power connections
  PI_PIN2 --> LC_INP
  PI_PIN6 --> LC_ING

  LC_OUTP --> RAD_VCC
  LC_OUTG --> RAD_GND

  LC_OUTP --> PRE_VDD
  LC_OUTG --> PRE_GND

  %% Signal connections
  RAD_IF --> CAP --> PRE_IN
  PRE_OUT --> TRS --> USB_MIC
  PRE_GND --> USB_GND

  %% USB data to Pi
  USB_USB --> PI_USBPORT

  %% Gain setup
  PRE_GAIN --> PRE_GND
  PRE_AR -. leave floating .- PRE_AR
```
