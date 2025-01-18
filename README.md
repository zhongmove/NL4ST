# INLAST
An interface for bridging the gap between natural language and spatio-temporal databases.
## Dependencies
   * torch-2.0.1 
   * Python-3.8
   * tomcat-8.5
   * Java-11
   * SECONDO
## Datasets is available in the Datastes folder.
## Usage
### Visit the online website
1. Integrate INLAST as an algebra into SECONDO database. (SECONDO: https://secondo-database.github.io/)
2. Online website is availabe at https://inlast.cpolar.top/nl2secondo/
### Deploy INLAST locally
1. Train the model to identify the type of NLQ.  
  `python LSTM/train.py`

2. Put the obtained models and related information in the directory INLAST/save_models.
   
3. Integrate INLAST as an algebra into SECONDO database. (SECONDO: https://secondo-database.github.io/)
   
4. Import the INLAST.war in Eclipse.
   
5. Publish the INLAST project to the tomcat server.  
   
6. Run the server.  
   
7. Open the database SECONDO.  
  `SecondoMonitor -s`

8. Type the url "http://localhost:8080/nl2secondo" in your browser.
