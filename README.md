# NL4ST
A Natural Language Query Tool for Spatio-temporal Databases

<img width="2868" height="3324" alt="fig2v3" src="https://github.com/user-attachments/assets/e1fe0de6-4568-4650-b1f5-577cac26a7a1" />




## Dependencies
   * torch-2.0.1 
   * Python-3.8
   * tomcat-8.5
   * Java-11
   * SECONDO
## Datasets are available in the Datastes folder
## Usage
### Visit the online website
1. Integrate NL4ST as an algebra into SECONDO database. (SECONDO: https://secondo-database.github.io/)
2. Online website is availabe at https://NL4ST.cpolar.top/nl2secondo/
### Deploy NL4ST locally
1. Train the model to identify the type of NLQ.  
  `python LSTM/train.py`

2. Put the obtained models and related information in the directory NL4ST/save_models.
   
3. Integrate NL4ST as an algebra into SECONDO database. (SECONDO: https://secondo-database.github.io/)
   
4. Import the NL4ST.war in Eclipse.
   
5. Publish the NL4ST project to the tomcat server.  
   
6. Run the server.  
   
7. Open the database SECONDO.  
  `SecondoMonitor -s`

8. Type the url "http://localhost:8080/nl2secondo/" in your browser.
