# NL4ST
A Natural Language Query Tool for Spatio-temporal Databases

![fig3 1](https://github.com/user-attachments/assets/661ad3b5-0a8e-4e30-bdd4-f3a3cf77b515)



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
