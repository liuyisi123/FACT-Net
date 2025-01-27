# **FACT-Net**  
### **A Two-Stage Fusion-CNN-Transformer Framework for ABP Signal Reconstruction in Cross-Platform Multi-Patient IoT Healthcare Systems**

---

## **Multimodal Physiological Signal Acquisition System**  
The multimodal physiological signal acquisition system designed in this study integrates advanced hardware modules to ensure precise, reliable, and efficient data acquisition for ABP reconstruction tasks.

### **System Components**  
1. **Hardware Modules**:  
   - **(a)** Sensor unit, signal processing circuitry, microcontroller unit (MCU), data processing module, and charging unit.  
   - **(b)** Simulation diagram illustrating the functional design of the acquisition system.  
   - **(c)** Front view of the acquisition device, emphasizing compactness and ergonomic design.  
   - **(d)** Rear view highlighting the modularity and scalability of system components.  
   - **(e)** High-accuracy PPG sensor for precise physiological signal collection.  
   - **(f)** 3D-printed casing integrated with a lithium battery, ensuring enhanced portability, durability, and usability.  

![System Overview](https://github.com/liuyisi123/FACT-Net/blob/main/Hardware.png)  

---

## **FACT-Net Architecture**  
FACT-Net adopts a two-stage hybrid architecture to achieve robust ABP waveform reconstruction, leveraging the strengths of CNNs and Transformers for cross-modal feature extraction and integration.

### **Stage I: Parallel Cross-Hybrid Architecture**  
- **Objective**: Extract multimodal features and provide constraint information.  
- **Components**:  
  - **Multi-Scale CNN Blocks**: Capture hierarchical features across varying temporal resolutions.  
  - **Mix-T Blocks**: Facilitate efficient multimodal feature fusion and integration.

### **Stage II: Serial Hybrid CNN-Transformer Structure**  
- **Objective**: Refine feature representation and achieve high-fidelity ABP waveform reconstruction.  
- **Components**:  
  - **Hybrid CNN Layers**: Enhance spatial and temporal feature extraction.  
  - **Transformer Modules**: Improve global context representation and signal reconstruction accuracy.

![FACT-Net Architecture](https://github.com/liuyisi123/FACT-Net/blob/main/FACT-Net.png)  

---

## **Cross-Platform Multi-Patient IoT Framework (CPMP-IoT)**  
The CPMP-IoT framework expands FACT-Net's application to real-world healthcare scenarios, offering scalable and efficient multi-patient health management solutions.

### **Framework Features**  
1. **Individual Monitoring APP**  
   - Supports **offline inference** for real-time, personalized health monitoring without reliance on cloud platforms.  
2. **Host Computer Integration**  
   - FACT-Net-enabled devices connect via **LAN**, providing seamless access to health reports and data through a secure web-based interface.  
3. **Ward-Level Multi-Patient Monitoring**  
   - Enables **simultaneous monitoring** of multiple patients in healthcare wards, ensuring scalability, reliability, and efficient resource management.

![CPMP-IoT Framework](https://github.com/liuyisi123/FACT-Net/blob/main/CPMP-IoT.png)  
