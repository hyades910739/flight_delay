# flight prediction


* scripts:
	* `feat_engineer.py`: functions to do the feat_engineer.  
	* `model.py`: the model class to predict.  
	* `run.py`: script to poc model on 9/1 split validation.  
	* `utli.py`: utility functions. 
	* `get_model.py`: run and save model.
	* `predict.py`: predict by model generated from `get_model.py`.  
	
* other files:
	* `data/airports.dat`: airport data from https://openflights.org/data.html.
	* `model_history/`: model logs.
	* `plots/`: some plot when EDA.
	* `models/`: place the model pickle file.
 
 * Requirements.
 To run the code, please install packages by:
 ```
pip install -r requirements.txt
 ```
 
 * to run model on training set and validate it.please run the following command. 
 	+ You can set upsampling rate by `-u`. 0 means no sampling. 
	+ set `-n True` if you want to use numeric features.
```
python run.py 
```

* to get the model (training on full dataset). Run following command with same arguments as above.
```
python get_model.py
```

* to run model on testing set. Run following command.
```
python predict.py -test_path 'path_to_your_test_file'
```

