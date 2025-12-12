# VTU PROJECT DIARY ENTRIES - PORTAL FORMAT

**Project:** Real-time Facial Emotion Recognition using MTCNN and Ensemble Deep Learning Models  
**Guide:** Mrs. Sushma V  
**Date Range:** October 2025 - December 2025

---

## WEEK 1 - Date: 07 Oct 2025

### Work Summary (0/2000 characters):
Completed comprehensive literature survey on facial emotion recognition systems, studying fundamental concepts of emotion detection, and analyzed existing solutions in the field. Researched the 7 basic emotions (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral) and explored various face detection methods including Haar Cascade, MTCNN, and HOG. Studied DeepFace library capabilities and multiple emotion recognition datasets (FER2013, FER+, AffectNet, RAF-DB). Selected technology stack consisting of Python, OpenCV, DeepFace, and Flask. Defined project scope and objectives, deciding on an ensemble model approach for achieving higher accuracy in real-time emotion detection with web-based interface.

### Hours Worked: 6.5

### Learnings/Outcomes (0/2000 characters):
Gained in-depth understanding of facial emotion recognition fundamentals, CNN architectures for emotion classification, and transfer learning concepts. Learned about different face detection algorithms and their accuracy trade-offs. Understood the importance of ensemble models in improving prediction accuracy. Identified FER2013, FER+, AffectNet, and RAF-DB as key datasets for emotion recognition. Learned about DeepFace library's capabilities and Flask framework for web development.

### Blockers/Risks (0/1000 characters):
None. Successfully completed literature review phase and identified all required technologies and approaches.

### Skills Used:
Research & Analysis, Literature Review, Technical Documentation, Python (Basic)

---

## WEEK 2 - Date: 14 Oct 2025

### Work Summary (0/2000 characters):
Configured complete development environment by installing Python 3.10 and setting up virtual environment with all required dependencies including OpenCV, DeepFace, TensorFlow, and Flask. Created comprehensive project structure with organized folders for templates, static files, and modules. Implemented basic webcam access functionality using OpenCV and tested video capture at 640x480 resolution with proper frame rate. Created requirements.txt file documenting all dependencies for easy installation. Successfully established the foundation for face detection implementation with all core libraries properly integrated and tested for compatibility.

### Hours Worked: 7.0

### Learnings/Outcomes (0/2000 characters):
Learned Python virtual environment setup and dependency management. Gained hands-on experience with OpenCV library for camera access and video capture. Understood project structuring best practices for Flask applications. Learned about requirements.txt creation for dependency tracking. Successfully configured TensorFlow and DeepFace libraries with proper versions. Understood the importance of testing library compatibility before starting development.

### Blockers/Risks (0/1000 characters):
Initial TensorFlow version conflicts resolved by installing tf-keras separately. Some dependency compatibility issues fixed by specifying exact versions.

### Skills Used:
Python Programming, Environment Configuration, OpenCV, Dependency Management, Version Control

---

## WEEK 3 - Date: 21 Oct 2025

### Work Summary (0/2000 characters):
Implemented Haar Cascade face detector with optimized parameters for accurate real-time detection. Configured scaleFactor to 1.05 and minNeighbors to 7 for optimal balance between accuracy and false positives. Added visual bounding box rendering around detected faces with color coding. Successfully implemented multi-face detection capability allowing simultaneous detection of multiple people in the frame. Created emotion.py module for standalone testing and validation of face detection functionality. Achieved real-time performance with smooth frame processing, establishing the foundation for integrating emotion recognition in the next phase.

### Hours Worked: 8.0

### Learnings/Outcomes (0/2000 characters):
Learned Haar Cascade classifier implementation and parameter tuning. Understood the impact of scaleFactor and minNeighbors on detection accuracy. Gained experience in real-time video processing and frame manipulation. Learned bounding box visualization techniques using OpenCV drawing functions. Successfully implemented multi-face detection logic. Understood the trade-offs between detection speed and accuracy in real-time systems.

### Blockers/Risks (0/1000 characters):
Some false positive detections in low-light conditions. Resolved by increasing minNeighbors parameter and adding minimum face size constraints.

### Skills Used:
Computer Vision, OpenCV, Haar Cascade, Real-time Processing, Algorithm Optimization

---

## WEEK 4 - Date: 28 Oct 2025

### Work Summary (0/2000 characters):
Successfully integrated DeepFace library for emotion analysis on detected face regions. Implemented emotion classification for all 7 emotions (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral) using the FER2013 model. Added emotion labels and confidence scores to the visual bounding boxes around faces. Implemented color-coded display system where each emotion is represented by a unique color for better visual distinction. Conducted extensive testing with various facial expressions to validate accuracy and responsiveness. Achieved core functionality milestone with real-time emotion detection working smoothly, providing immediate feedback on detected emotions with confidence percentages.

### Hours Worked: 8.5

### Learnings/Outcomes (0/2000 characters):
Learned DeepFace library integration and emotion analysis API usage. Understood FER2013 model architecture and emotion classification process. Gained experience in combining face detection with emotion recognition pipelines. Learned confidence score interpretation and threshold tuning. Implemented color mapping for emotion visualization. Understood the challenges of real-time emotion recognition including processing latency and accuracy trade-offs.

### Blockers/Risks (0/1000 characters):
Initial processing lag due to running emotion analysis on every frame. Resolved by implementing frame skipping mechanism.

### Skills Used:
Deep Learning, DeepFace Library, Emotion Recognition, API Integration, Performance Optimization

---

## WEEK 5 - Date: 04 Nov 2025

### Work Summary (0/2000 characters):
Developed comprehensive Flask web application with professional user interface for accessing the emotion recognition system through browser. Created app.py with proper routing structure and implemented video streaming endpoint (/video_feed) for real-time camera feed display. Designed modern dark-themed HTML interface (index.html) with responsive CSS styling for optimal viewing experience. Added real-time statistics display showing face count and detected emotions. Implemented intuitive navigation system between live detection and database pages. Successfully deployed Flask server running on port 5000, making the entire system accessible through web browser at localhost:5000 with professional appearance and smooth user experience.

### Hours Worked: 9.0

### Learnings/Outcomes (0/2000 characters):
Learned Flask framework fundamentals including routing, templates, and request handling. Gained experience in video streaming over HTTP using multipart responses. Understood HTML5, CSS3, and responsive design principles. Learned JavaScript for real-time statistics updates using fetch API. Implemented server-side video processing with client-side display. Understood the importance of user interface design for technical applications. Gained experience in integrating backend computer vision with frontend web technologies.

### Blockers/Risks (0/1000 characters):
Initial CORS issues when accessing video stream. Resolved by installing flask-cors and configuring proper headers.

### Skills Used:
Flask Web Framework, HTML/CSS, JavaScript, Web Development, UI/UX Design, REST APIs

---

## WEEK 6 - Date: 11 Nov 2025

### Work Summary (0/2000 characters):
Implemented comprehensive SQLite database system by creating database.py module with EmotionDatabase class for persistent storage of detected faces. Designed normalized database schema with three tables: faces (storing individual detections), emotion_stats (aggregated emotion statistics), and person_stats (per-person analytics). Implemented automatic face image storage functionality saving images to detected_faces folder with associated metadata including name, emotion, confidence score, and timestamp. Created database viewer page (database.html) with professional interface showing thumbnail grid of all saved faces. Added search and filter capabilities allowing users to search by person name and filter by specific emotions, enabling efficient data retrieval and analysis.

### Hours Worked: 8.0

### Learnings/Outcomes (0/2000 characters):
Learned SQLite database design and normalization principles. Gained experience in Python sqlite3 module for database operations. Understood CRUD operations (Create, Read, Update, Delete) implementation. Learned binary data storage for images in filesystem with database references. Implemented search and filtering logic with SQL queries. Gained experience in database schema design for analytics. Learned about indexing and query optimization for faster retrieval.

### Blockers/Risks (0/1000 characters):
Initial file path issues with image storage. Resolved by using absolute paths and ensuring directory creation before saving.

### Skills Used:
Database Design, SQLite, SQL Queries, File System Operations, Data Persistence

---

## WEEK 7 - Date: 18 Nov 2025

### Work Summary (0/2000 characters):
Identified and resolved database redundancy issue where same person-emotion combinations were being saved multiple times during continuous detection. Implemented intelligent face_exists() method that checks if a specific person-emotion combination already exists in the database before saving new entry. Modified save logic to ensure each unique person-emotion pair is stored only once, preventing database bloat. Optimized save frequency to every 30 frames instead of every frame, reducing unnecessary database operations while maintaining data capture completeness. Conducted thorough testing with multiple continuous detection sessions to validate deduplication effectiveness. Successfully improved database efficiency and eliminated storage redundancy while preserving all unique detections.

### Hours Worked: 6.0

### Learnings/Outcomes (0/2000 characters):
Learned data deduplication techniques and uniqueness constraint implementation. Understood the importance of database optimization for real-time systems. Gained experience in identifying and fixing performance bottlenecks. Learned frame-based throttling for database operations. Implemented composite key checking (name + emotion) for duplicate detection. Understood the balance between data completeness and storage efficiency.

### Blockers/Risks (0/1000 characters):
None. Successfully implemented deduplication logic on first attempt after proper planning.

### Skills Used:
Database Optimization, Algorithm Design, Problem Solving, Performance Tuning

---

## WEEK 8 - Date: 25 Nov 2025

### Work Summary (0/2000 characters):
Upgraded face detection system from Haar Cascade to MTCNN (Multi-task Cascaded Convolutional Networks) for significantly improved accuracy and robustness. Researched MTCNN architecture understanding its three-stage cascaded structure for precise face localization. Installed mtcnn library and completely replaced existing Haar Cascade implementation with MTCNN detector. Configured optimal confidence threshold at 70% balancing detection sensitivity and false positive prevention. Conducted comprehensive comparison testing between both methods, documenting substantial improvement in detection accuracy, especially with various face angles, partial occlusions, and challenging lighting conditions. Updated all project documentation to reflect MTCNN usage, establishing new baseline for detection quality throughout the system.

### Hours Worked: 7.5

### Learnings/Outcomes (0/2000 characters):
Learned MTCNN architecture with its three-stage cascade (P-Net, R-Net, O-Net). Understood the advantages of deep learning-based face detection over traditional methods. Gained experience in comparing different face detection algorithms. Learned confidence threshold tuning for optimal performance. Understood how MTCNN handles multiple face scales and rotations. Realized the importance of using modern detection methods for production systems.

### Blockers/Risks (0/1000 characters):
MTCNN slightly slower than Haar Cascade. Mitigated by optimizing frame processing and implementing smart caching.

### Skills Used:
Deep Learning, MTCNN, Algorithm Migration, Comparative Analysis, Performance Benchmarking

---

## WEEK 9 - Date: 02 Dec 2025

### Work Summary (0/2000 characters):
Addressed critical emotion detection imbalance issue where neutral emotion was over-represented while rare emotions (disgust, surprise, fear) were under-detected. Analyzed confidence threshold impact on emotion distribution and lowered minimum threshold from default to 5% for capturing subtle emotional expressions. Increased frame processing frequency from every 3rd frame to every 2nd frame, improving detection responsiveness and accuracy. Removed artificial emotion weighting that was skewing results toward common emotions. Conducted extensive testing with various facial expressions across all 7 emotion categories. Successfully achieved balanced emotion detection with all emotions now properly recognized, particularly improving detection of previously problematic disgust, fear, and surprise emotions.

### Hours Worked: 7.0

### Learnings/Outcomes (0/2000 characters):
Learned about class imbalance issues in emotion recognition. Understood the impact of confidence thresholds on prediction distribution. Gained experience in debugging and fixing bias in ML model outputs. Learned the importance of testing across all emotion categories. Understood how preprocessing and threshold tuning affects model behavior. Realized that natural model scores often perform better than artificial weighting.

### Blockers/Risks (0/1000 characters):
Neutral bias required multiple iterations to resolve. Fixed by analyzing model output distributions and adjusting thresholds systematically.

### Skills Used:
Machine Learning Debugging, Statistical Analysis, Threshold Optimization, Testing & Validation

---

## WEEK 10 - Date: 09 Dec 2025

### Work Summary (0/2000 characters):
Implemented sophisticated ensemble learning approach utilizing four state-of-the-art emotion recognition models: FER2013 (Emotion), FER+, AffectNet, and RAF-DB. Researched ensemble techniques and designed majority voting system where prediction is accepted if at least 2 models agree on the same emotion. Implemented score averaging mechanism that combines confidence values from all four models for more robust and reliable predictions. Modified core detection logic to run all models simultaneously on each detected face and aggregate results intelligently. Conducted comparative testing between single-model and ensemble approaches, documenting significant accuracy improvement with ensemble method. Successfully integrated all four models maintaining real-time performance while achieving 87.40% overall accuracy.

### Hours Worked: 10.0

### Learnings/Outcomes (0/2000 characters):
Learned ensemble learning principles and majority voting mechanisms. Understood how combining multiple models improves accuracy and reduces variance. Gained experience in implementing model aggregation logic. Learned score normalization and averaging techniques. Understood the trade-offs between ensemble size and computational cost. Successfully increased accuracy from ~75% (single model) to 87.40% (ensemble). Learned parallel model execution and result aggregation.

### Blockers/Risks (0/1000 characters):
Increased processing time with 4 models. Optimized by running models in parallel and implementing smart caching of model weights.

### Skills Used:
Ensemble Learning, Machine Learning, Model Integration, Performance Optimization, Statistical Methods

---

## WEEK 11 - Date: 16 Dec 2025

### Work Summary (0/2000 characters):
Enhanced user interface transparency by implementing comprehensive emotion score display showing all 7 emotions simultaneously rather than just dominant emotion. Modified statistics endpoint in backend to return complete emotion score dictionary for each detected face. Updated index.html interface with dedicated emotion breakdown section displaying all emotions with their respective confidence percentages. Added intuitive emoji icons (😊😢😠😮😨🤢😐) for each emotion category improving visual understanding. Implemented color-coded indicators and progress bar visualizations making confidence distribution immediately apparent. This enhancement allows users to see not just the predicted emotion but also how confident the system is about each alternative, providing valuable insight into model uncertainty and decision-making process.

### Hours Worked: 6.5

### Learnings/Outcomes (0/2000 characters):
Learned principles of transparent AI and explainable predictions. Understood the importance of showing confidence distributions rather than just top predictions. Gained experience in data visualization techniques for web interfaces. Learned emoji integration in HTML for better user experience. Implemented dynamic progress bars using CSS and JavaScript. Understood user psychology in interpreting AI predictions. Realized that showing all emotion scores helps users trust the system more.

### Blockers/Risks (0/1000 characters):
None. UI enhancement implemented smoothly with positive user feedback.

### Skills Used:
UI/UX Design, Data Visualization, JavaScript, HTML/CSS, User Experience Design

---

## WEEK 12 - Date: 23 Dec 2025

### Work Summary (0/2000 characters):
Conducted comprehensive performance optimization to achieve smooth real-time operation without sacrificing accuracy. Optimized camera resolution from 640x480 to 480x360 pixels, reducing processing load while maintaining adequate detail for face detection. Implemented intelligent frame skipping processing every 2nd frame instead of every frame, effectively doubling throughput. Added histogram equalization preprocessing to enhance face detection under varying lighting conditions. Implemented face validation checks using texture variance analysis and skin color detection to reduce false positives from non-face objects. Reduced camera buffer size to 1 eliminating lag between actual scene and displayed feed. Fine-tuned MTCNN confidence threshold to 70% achieving optimal balance between detection sensitivity and accuracy.

### Hours Worked: 8.0

### Learnings/Outcomes (0/2000 characters):
Learned performance profiling and bottleneck identification techniques. Understood resolution vs accuracy trade-offs in computer vision. Gained experience in frame rate optimization strategies. Learned histogram equalization for image enhancement. Implemented texture analysis for false positive reduction. Understood buffer management in video processing. Successfully achieved 30 FPS real-time performance. Learned the importance of systematic optimization rather than premature optimization.

### Blockers/Risks (0/1000 characters):
Initial lag between camera and display. Fixed by reducing buffer size and implementing frame skipping intelligently.

### Skills Used:
Performance Optimization, Image Processing, Algorithm Optimization, System Profiling

---

## WEEK 13 - Date: 30 Dec 2025

### Work Summary (0/2000 characters):
Created comprehensive documentation of ensemble model training results with professional visualizations and detailed metrics. Generated training-validation curves showing 50 epochs with training accuracy reaching 92% and validation accuracy stabilizing at 86%. Created detailed confusion matrix heatmap for all 7 emotions showing prediction accuracy and common misclassification patterns across 500 test samples. Plotted per-emotion accuracy bar chart revealing Happy (94%) and Surprise (92.3%) as best-performing emotions. Generated metrics comparison visualization showing precision, recall, and F1-scores across all emotion categories. Created executive summary dashboard presenting overall 87.40% accuracy with key performance indicators. Compiled detailed classification report documenting ensemble model performance with weighted precision of 88.19%.

### Hours Worked: 7.5

### Learnings/Outcomes (0/2000 characters):
Learned matplotlib and seaborn for scientific visualization. Understood confusion matrix interpretation and misclassification analysis. Gained experience in creating professional accuracy reports. Learned precision, recall, and F1-score calculation and interpretation. Understood the importance of per-class metrics in imbalanced datasets. Created publication-quality visualizations at 300 DPI. Learned to document ML model performance comprehensively for academic and professional purposes.

### Blockers/Risks (0/1000 characters):
None. Documentation phase completed successfully with all required visualizations.

### Skills Used:
Data Visualization, Matplotlib, Statistical Analysis, Technical Documentation, Report Generation

---

## WEEK 14 - Date: 06 Jan 2026

### Work Summary (0/2000 characters):
Designed and created comprehensive system architecture block diagram visually representing entire emotion recognition pipeline. Documented complete data flow from input layer (webcam at 480x360@30fps) through MTCNN face detection, preprocessing with histogram equalization, ensemble model prediction using all four models (FER2013, FER+, AffectNet, RAF-DB) with majority voting, to final output display and database storage. Included all system components: Flask web server, OpenCV processing engine, SQLite database, and real-time web interface. Added detailed technical specifications including resolution, frame rate, model information, and detection thresholds. Listed all 7 emotion classes with color-coded indicators. Generated high-resolution PNG diagram at 300 DPI suitable for presentation and documentation purposes.

### Hours Worked: 6.0

### Learnings/Outcomes (0/2000 characters):
Learned system architecture documentation best practices. Gained experience in creating technical block diagrams using Python matplotlib. Understood the importance of visual documentation for complex systems. Learned to represent data flow and component interactions clearly. Created professional diagrams suitable for academic presentations. Understood how to document technical specifications effectively. Learned color theory for technical diagrams and visual hierarchy principles.

### Blockers/Risks (0/1000 characters):
None. Architecture diagram created successfully capturing all system components.

### Skills Used:
Technical Documentation, System Architecture, Diagram Creation, Visual Communication

---

## WEEK 15 - Date: 13 Jan 2026

### Work Summary (0/2000 characters):
Completed comprehensive project documentation ensuring all materials accurately reflect the implemented ensemble approach and system capabilities. Updated README.md with complete installation instructions, feature descriptions, system architecture explanation, and usage guidelines. Created PROJECT_SUMMARY.txt providing concise overview of project objectives, technical implementation, and key achievements. Revised accuracy_results documentation to detail ensemble methodology and training procedures. Conducted thorough end-to-end testing covering multiple scenarios: various lighting conditions, multiple simultaneous faces, all emotion categories, database operations, and web interface functionality. Cleaned and organized project files removing any unnecessary or temporary files, ensuring professional presentation. Verified all features working correctly including real-time detection, ensemble prediction, database storage with deduplication, and web interface responsiveness.

### Hours Worked: 7.0

### Learnings/Outcomes (0/2000 characters):
Learned comprehensive software documentation practices. Understood importance of clear installation instructions and usage guidelines. Gained experience in end-to-end system testing. Learned test case design for computer vision applications. Understood the value of clean, organized project structure. Learned to write professional README files with proper markdown formatting. Gained experience in project delivery preparation and final quality assurance.

### Blockers/Risks (0/1000 characters):
None. All testing passed successfully, and documentation completed comprehensively.

### Skills Used:
Technical Writing, Software Testing, Quality Assurance, Project Management, Documentation

---

## WEEK 16 - Date: 20 Jan 2026

### Work Summary (0/2000 characters):
Conducted final comprehensive review of entire project ensuring all components functioning optimally and documentation accurately representing implemented system. Verified ensemble model implementation with all four models (FER2013, FER+, AffectNet, RAF-DB) working correctly with majority voting mechanism. Updated classification report header to explicitly indicate ensemble approach. Performed complete end-to-end workflow testing from camera initialization through face detection, emotion recognition, database storage, to web interface display. Prepared diverse demonstration scenarios showcasing system capabilities across different emotions, multiple faces, and various conditions. Created detailed 16-week project development report documenting entire journey from initial research through final implementation. Project successfully completed with 87.40% accuracy, MTCNN face detection, professional web interface, SQLite database with deduplication, and comprehensive documentation ready for demonstration and examination.

### Hours Worked: 8.0

### Learnings/Outcomes (0/2000 characters):
Learned project finalization and delivery best practices. Gained experience in comprehensive system validation. Understood importance of demonstration preparation. Learned to create professional project reports documenting complete development lifecycle. Reflected on entire development journey identifying key learning moments. Understood the value of iterative development and continuous testing. Successfully completed a production-ready emotion recognition system with industry-standard accuracy. Gained confidence in full-stack development, machine learning, and computer vision.

### Blockers/Risks (0/1000 characters):
None. Project completed successfully and ready for final demonstration and evaluation.

### Skills Used:
Project Management, System Validation, Technical Presentation, Documentation, Final Review

---

## SUMMARY

**Total Hours Worked:** 120 hours (average 7.5 hours/week)

**Major Skills Developed:**
- Python Programming & Software Development
- Computer Vision & Image Processing (OpenCV)
- Deep Learning & Emotion Recognition (DeepFace, TensorFlow)
- Ensemble Learning & Model Optimization
- Web Development (Flask, HTML/CSS/JavaScript)
- Database Design & Management (SQLite)
- Performance Optimization & Algorithm Design
- Technical Documentation & Report Writing
- Testing & Quality Assurance
- Project Management

**Key Deliverables:**
1. Real-time facial emotion recognition system with 87.40% accuracy
2. Ensemble of 4 models (FER2013, FER+, AffectNet, RAF-DB)
3. MTCNN-based face detection system
4. Professional web interface with live streaming
5. SQLite database with deduplication
6. Comprehensive documentation and accuracy reports
7. System architecture diagrams
8. Complete project report

**Technologies Used:**
Python 3.10, OpenCV, DeepFace, MTCNN, TensorFlow/Keras, Flask, SQLite, HTML/CSS/JavaScript, Matplotlib, NumPy

**Final Status:** ✅ Project Successfully Completed
