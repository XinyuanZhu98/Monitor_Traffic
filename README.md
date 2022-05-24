# Monitor_Traffic
Monitor traffic in Singapore by extracting images from a website and performing object detection.  
The image source: https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras.html

To get started, install the required dependencies by running
```python
pip install -r requirements.txt
```

* **Task 1**: Extract traffic images from a given AREA or all available areas.
* **Task 2**: Detect and count the number of cars (or other classes of objects as specified) in the extracted images.
The two tasks can be done by executing the following.
```python
python main.py
```
You can also specify your own settings by running
```python
python main.py --area=your_area 
               --output_dir==path_to_your_output_folder 
               --model_quality=your_model_quality 
               --confidence=your_confidence 
               --target_class=your_coco_classes_for_detection
```
The model that yields more accurate prediction results will take a longer time to detect. Once you have chosen your own model, it will be automatically downloaded from the torchvision module if not yet exist. And note that sometimes some images on the website are cropped or damaged and these "bad" images will be ignored and skipped by the program.

* **Task 3**: Perform a near real-time monitoring of the traffic.  
This is done by triggering a DAG in Apache Airflow every three minutes which I think is sufficient since the images on the given website refresh roughly at the same time interval (3 min) through observation:). By default, only cars are monitored. The code can be modified to include other vehicle classes such as "truck", but as the processing time should not exceed three minutes, I stick with cars.   
Since this task is done by the Airflow DAG (```dag_id = monitor```) composed of two sub-tasks ```collect``` and ```detect``` using ```BashOperator``` included in monitor.py, please move the Python script monitor.py to your AIRFLOW_HOME/dags/. Note that the DAG will need to collaborate with the other two scripts collect.py and detect.py. Please create a new folder helpers/ under AIRFLOW_HOME/dags/ and place both scripts in AIRFLOW_HOME/dags/helpers/, shown as follows. By default, the collected images will be saved to AIRFLOW_HOME/images/.  

```python
|-- AIRFLOW_HOME
    |-- dags
    |   |-- monitor.py
    |   |-- helpers
    |   |   |-- collect.py
    |   |   |-- detect.py
```
**Logging:** By default, logs are written to the .log files. However, it is possible to enable custom configuration and log to the console by
setting 
```python
LOGGING_CONFIG["handlers"]["task"] = {
    "class": "logging.StreamHandler",
    "formatter": "airflow",
    "stream": sys.stdout,
}
```
More info on custom logging:  
https://stackoverflow.com/questions/68467728/custom-logging-in-airflow;  
https://airflow.apache.org/docs/apache-airflow/stable/logging-monitoring/logging-tasks.html#advanced-configuration
