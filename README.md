# INLAST
An interface for bridging the gap between natural language and spatio-temporal databases.
## Dependencies
   * torch-2.0.1 
   * Python-3.8
   * tomcat-8.0
   * Java-11
   * SECONDO
## Usage
1. Train the model to identify the type of NLQ.  
  `python LSTM/train.py`

2. Put the obtained models and related information in the directory INLAST/save_models.
   
3. Integrate INLAST as an algebra into SECONDO database. (SECONDO: https://secondo-database.github.io/).
   
4. Import the INLAST.war in Eclipse.
   
5. Publish the project to the tomcat server.
   
6. Open the dataabse SECONDO.  
  `SecondoMonitor -s`

7. Type the url "http://localhost:8080/INLAST" in your browser.
