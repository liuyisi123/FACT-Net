# **FACT-Net**  
### **A Two-Stage Fusion-CNN-Transformer Framework for ABP Signal Reconstruction in Cross-Platform Multi-Patient IoT Healthcare Systems**

---

## **🧠 Multimodal Physiological Signal Acquisition System**  
The **multimodal physiological signal acquisition system** developed in this study integrates cutting-edge hardware components, ensuring precise, reliable, and efficient data acquisition for ABP reconstruction tasks in diverse healthcare settings.

### **🔧 System Components**  
1. **Hardware Modules**:  
   - **(a)** 🛠 **Sensor Unit**: Collects physiological signals (e.g., ECG, PPG).  
   - **(b)** 📊 **Signal Processing Circuitry**: Processes raw signals into usable data.  
   - **(c)** 🖥 **Microcontroller Unit (MCU)**: Coordinates the system’s operations.  
   - **(d)** 💾 **Data Processing Module**: Handles complex computations for signal analysis.  
   - **(e)** 🔋 **Charging Unit**: Powers the system with high-efficiency charging.  
2. **System Design**:  
   - **(f)** 🖼 **Simulation Diagram**: Illustrates the functional design of the system.  
   - **(g)** 🔲 **Front View**: Showcases the compact and ergonomic design.  
   - **(h)** 🔲 **Rear View**: Highlights modularity for easy upgrades.  
   - **(i)** 🌟 **PPG Sensor**: Provides high-accuracy signal acquisition.  
   - **(j)** 🔋 **3D-Printed Casing**: Integrated with a lithium battery for enhanced portability and durability.  

![System Overview](https://github.com/liuyisi123/FACT-Net/blob/main/Hardware.png)  

---

## **🔍 FACT-Net Architecture**  
FACT-Net leverages a **two-stage hybrid architecture** combining CNNs and Transformers to achieve high-fidelity ABP waveform reconstruction. This architecture enables robust **cross-modal feature extraction** and **integration** for optimal performance.

### **🚀 Stage I: Parallel Cross-Hybrid Architecture**  
- **Objective**: Efficiently extract multimodal features and provide constraint information for accurate ABP reconstruction.  
- **Components**:  
  - **🔲 Multi-Scale CNN Blocks**: Capture hierarchical features across varying temporal resolutions.  
  - **🔀 Mix-T Blocks**: Facilitate efficient multimodal feature fusion for improved integration.

### **🔨 Stage II: Serial Hybrid CNN-Transformer Structure**  
- **Objective**: Refine feature representations and ensure high-fidelity ABP waveform reconstruction.  
- **Components**:  
  - **🌐 Hybrid CNN Layers**: Enhance spatial and temporal feature extraction.  
  - **⚡ Transformer Modules**: Improve global context representation for superior signal reconstruction accuracy.

![FACT-Net Architecture](https://github.com/liuyisi123/FACT-Net/blob/main/FACT-Net.png)  

---

## **🌐 Cross-Platform Multi-Patient IoT Framework (CPMP-IoT)**  
The **CPMP-IoT framework** extends FACT-Net’s capabilities to real-world healthcare applications, offering a scalable and reliable solution for **multi-patient health management**.

### **🔑 Framework Features**  
1. **📱 Individual Monitoring APP**  
   - **Offline Inference**: Real-time, personalized health monitoring without cloud reliance.  
2. **💻 Host Computer Integration**  
   - Devices connect via **LAN**, enabling seamless access to health data and reports through a secure web interface.  
3. **🏥 Ward-Level Multi-Patient Monitoring**  
   - **Simultaneous Monitoring**: Enables the management of multiple patients in healthcare wards, ensuring scalability and efficient resource use.

![CPMP-IoT Framework](https://github.com/liuyisi123/FACT-Net/blob/main/CPMP-IoT.png)  

---

## **📑 Appendices**  

### **📐 Appendix I: Circuit Schematic**  
The **circuit schematic** provides a comprehensive illustration of the hardware design, detailing the interconnections between key components in the physiological signal acquisition system.  
[**Download Circuit Schematic PDF**](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-II-Circuit%20Schematic.pdf)  

### **🖥 Appendix II: PCB Design**  
The **PCB design** outlines the printed circuit board layout, ensuring optimal integration and functionality of the system components.  
[**Download PCB Design PDF**](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-III-PCB.pdf)

---

## **💾 Download Buttons**  
You can easily download the **circuit schematic** and **PCB design** from the following links:

[![Download Circuit Schematic](https://img.shields.io/badge/Download%20Circuit%20Schematic-blue?style=for-the-badge&logo=github)](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-II-Circuit%20Schematic.pdf)  
[![Download PCB Design](https://img.shields.io/badge/Download%20PCB%20Design-blue?style=for-the-badge&logo=github)](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-III-PCB.pdf)
