import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from inz import *

test_size = np.nan
c, mn, si, cr, ni, mo, p, s, cu, v = float, float, float, float, float, float, float, float, float, float
temperature, speed = float, float
#reo_var = float

predictDataframe = pd.DataFrame()

with open('model_speed.pkl','rb') as model:
    dec_tree = pickle.load(model)
    error_tree = pickle.load(model)


data_speed = pd.read_excel("dane.xls", sheet_name="Speed") 
X, Y = data_speed.drop(data_speed.columns[-1], axis = 1), data_speed[data_speed.columns[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

print(mae(Y_train, dec_tree.predict(X_train)))
print(mse(Y_train, dec_tree.predict(X_train)))
print(rmse(Y_train, dec_tree.predict(X_train)))
    
print(mae(Y_test, dec_tree.predict(X_test)))
print(mse(Y_test, dec_tree.predict(X_test)))
print(rmse(Y_test, dec_tree.predict(X_test))) 

#Functions
def search_file():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetype=(("xls files", "*.xls"),("All Files", "*.*")))
    labelFile["text"] = filename
    return None

def load_data():
    path = labelFile["text"]
    try:
        filename = r"{}".format(path)
        if filename[-4:] == ".xls":
            dataFrame = pd.read_excel(filename)
        elif filename[-4:] == ".csv":
            dataFrame = pd.read_csv(filename)
        else:
            tk.messagebox.showerror("Error", f"Invalid extension of file")
            return None
        
    except FileNotFoundError:
        tk.messagebox.showerror("Error", f"No such file as {path}")
        return None
    except ValueError:
        tk.messagebox.showerror("Error", f"Invalid file")
        return None
    
    global dataset
    dataset = dataFrame
    
    clear_table()
    treeView["column"] = list(dataFrame.columns)
    treeView["show"] = "headings"
    for column in treeView["columns"]:
        treeView.heading(column, text=column)
        treeView.column(column, anchor="center", minwidth=75, width=100, stretch=True)
    dataFrameRows = dataFrame.to_numpy().tolist()
    for row in dataFrameRows:
        treeView.insert("", "end", values=row)
    
    return None

def clear_table():
    treeView.delete(*treeView.get_children())
    return None

def fit():

    try:
    
        test_size = testSizeEntry.get()
    
        X, Y = dataset.drop(dataset.columns[-1], axis = 1), dataset[dataset.columns[-1]]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = float(test_size), random_state = 0)

        dec_tree = DecisionTree(max_depth=100)
        dec_tree.fit(X, Y)
    
        mae_train = mae(Y_train, dec_tree.predict(X_train))
        mse_train = mse(Y_train, dec_tree.predict(X_train))
        rmse_train = rmse(Y_train, dec_tree.predict(X_train))
    
        mae_test = mae(Y_test, dec_tree.predict(X_test))
        mse_test = mse(Y_test, dec_tree.predict(X_test))
        rmse_test = rmse(Y_test, dec_tree.predict(X_test)) 
    
        maeTrain = tk.Label(trainingLabel, text="MAE = {}".format(round(mae_train, 5)))
        maeTrain.place(relx=0, rely=0)
        mseTrain = tk.Label(trainingLabel, text="MSE = {}".format(round(mse_train, 5)))
        mseTrain.place(relx=0, rely=0.2)
        rmseTrain = tk.Label(trainingLabel, text="RMSE = {}".format(round(rmse_train,5)))
        rmseTrain.place(relx=0, rely=0.4)
        
        maeTest = tk.Label(testLabel, text="MAE = {}".format(round(mae_test,5)))
        maeTest.place(relx=0, rely=0)
        mseTest = tk.Label(testLabel, text="MSE = {}".format(round(mse_test,5)))
        mseTest.place(relx=0, rely=0.2)
        rmseTest = tk.Label(testLabel, text="RMSE = {}".format(round(rmse_test,5)))
        rmseTest.place(relx=0, rely=0.4) 
        
    
    except NameError:
        tk.messagebox.showerror("Error", f"Load dataset first")
        return None
    
    tk.messagebox("MAE,MSE & RMSE are used to check how close estimates or forecasts are to actual values. Lower the errors, the closer is forecast to actual.")
     

def predict():
    c = cEntry.get()
    mn = mnEntry.get()
    si = siEntry.get()
    cr = crEntry.get()
    ni = niEntry.get()
    mo = moEntry.get()
    p = pEntry.get()
    s = sEntry.get()
    cu = cuEntry.get()
    v = vEntry.get()
    temperature = tempEntry.get()
    speed = speedEntry.get()
    
    #reo_var = float(reoEntry.get())
    #dict = [{'C':float(c), 'Mn':float(mn), 'Si':float(si), 'Cr':float(cr), 'Ni':float(ni), 'Mo':float(mo), 
    #        'P':float(p), 'S':float(s), 'Cu':float(cu), 'V':float(v), 'Speed':None, 'Temperature':float(temperature), 
    #        'Shear Rate':None, 'Shear Stress':None, 'Torque':None}]

    dict = [{'C':float(c), 'Mn':float(mn), 'Si':float(si), 'Cr':float(cr), 'Ni':float(ni), 'Mo':float(mo), 
            'P':float(p), 'S':float(s), 'Cu':float(cu), 'V':float(v), 'Temperature':float(temperature), 'Speed':float(speed)}]
    
    '''match reologyList.current():
        case 0:
            dict[0]['Shear Rate'] = reo_var
        case 1:
            dict[0]['Shear Stress'] = reo_var
        case 2:
            dict[0]['Speed'] = reo_var  
        case 3:
            dict[0]['Torque'] = reo_var'''
    df_prediction = pd.DataFrame(dict)
    #print(dec_tree.predict(df_prediction))          
    tk.messagebox.showinfo("Viscosity.",  "Predicted viscosity is {}".format(dec_tree.predict(df_prediction)))

root = tk.Tk()

root.title("Reology App")
root.geometry("700x500")
#root.pack_propagate(False) 
root.resizable(False, False)

#MAE MSE RMSE
trainingLabel = tk.LabelFrame(root, text="Training")
trainingLabel.place(height=100, width=250, rely=0, relx=0)

testLabel = tk.LabelFrame(root, text="Test")
testLabel.place(height=100, width=250, rely=0.2, relx=0)

maeTrain = tk.Label(trainingLabel, text="MAE = {}".format(round(error_tree.mae_train,5)))
maeTrain.place(relx=0, rely=0)
mseTrain = tk.Label(trainingLabel, text="MSE = {}".format(round(error_tree.mse_train,5)))
mseTrain.place(relx=0, rely=0.2)
rmseTrain = tk.Label(trainingLabel, text="RMSE = {}".format(round(error_tree.rmse_train,5)))
rmseTrain.place(relx=0, rely=0.4)
    
maeTest = tk.Label(testLabel, text="MAE = {}".format(round(error_tree.mae_test,5)))
maeTest.place(relx=0, rely=0)
mseTest = tk.Label(testLabel, text="MSE = {}".format(round(error_tree.mse_test,5)))
mseTest.place(relx=0, rely=0.2)
rmseTest = tk.Label(testLabel, text="RMSE = {}".format(round(error_tree.rmse_test,5)))
rmseTest.place(relx=0, rely=0.4)  

#Predict
predictLabel = tk.LabelFrame(root, text="Prediction")
predictLabel.place(height=291, width=250, rely=0.41, relx=0)

chem_desc = tk.Label(predictLabel, text="Chemical composition")
chem_desc.place(relx=0, rely=0)

cLabel = tk.Label(predictLabel, text="C")
cLabel.place(relx=0, rely=0.1)

cEntry = tk.Entry(predictLabel, textvariable=c, width=5)
cEntry.place(relx=0.05, rely=0.1)
cEntry.insert(0, 0.0)

mnLabel = tk.Label(predictLabel, text="Mn")
mnLabel.place(relx=0.2, rely=0.1)

mnEntry = tk.Entry(predictLabel, textvariable=mn, width=5)
mnEntry.place(relx=0.3, rely=0.1)
mnEntry.insert(0, 0.0)

siLabel = tk.Label(predictLabel, text="Si")
siLabel.place(relx=.45, rely=0.1)

siEntry = tk.Entry(predictLabel, textvariable=si, width=5)
siEntry.place(relx=.5, rely=0.1)
siEntry.insert(0, 0.0)

crLabel = tk.Label(predictLabel, text="Cr")
crLabel.place(relx=.65, rely=0.1)

crEntry = tk.Entry(predictLabel, textvariable=cr, width=5)
crEntry.place(relx=.71, rely=0.1)
crEntry.insert(0, 0.0)

pLabel = tk.Label(predictLabel, text="P")
pLabel.place(relx=0, rely=0.2)

pEntry = tk.Entry(predictLabel, textvariable=p, width=5)
pEntry.place(relx=0.05, rely=0.2)
pEntry.insert(0, 0.0)

moLabel = tk.Label(predictLabel, text="Mo")
moLabel.place(relx=0.2, rely=0.2)

moEntry = tk.Entry(predictLabel, textvariable=mo, width=5)
moEntry.place(relx=0.3, rely=0.2)
moEntry.insert(0, 0.0)

niLabel = tk.Label(predictLabel, text="Ni")
niLabel.place(relx=.45, rely=0.2)

niEntry = tk.Entry(predictLabel, textvariable=ni, width=5)
niEntry.place(relx=.5, rely=0.2)
niEntry.insert(0, 0.0)

sLabel = tk.Label(predictLabel, text="S")
sLabel.place(relx=.65, rely=0.2)

sEntry = tk.Entry(predictLabel, textvariable=s, width=5)
sEntry.place(relx=.71, rely=0.2)
sEntry.insert(0, 0.0)

cuLabel = tk.Label(predictLabel, text="Cu")
cuLabel.place(relx=.0, rely=0.3)

cuEntry = tk.Entry(predictLabel, textvariable=cu, width=5)
cuEntry.place(relx=.05, rely=0.3)
cuEntry.insert(0, 0.0)

vLabel = tk.Label(predictLabel, text="V")
vLabel.place(relx=.24, rely=0.3)

vEntry = tk.Entry(predictLabel, textvariable=v, width=5)
vEntry.place(relx=.3, rely=0.3)
vEntry.insert(0, 0.0)

#chem_desc = tk.Label(predictLabel, text="Rheological properties")
#chem_desc.place(relx=0, rely=.4)

tempLabel = tk.Label(predictLabel, text="Temperature")
tempLabel.place(relx=.0, rely=0.5)

tempEntry = tk.Entry(predictLabel, textvariable=temperature, width=6)
tempEntry.place(relx=.35, rely=0.5)
tempEntry.insert(0, 0.0)

speedLabel = tk.Label(predictLabel, text="Speed")
speedLabel.place(relx=.0, rely=0.6)

speedEntry = tk.Entry(predictLabel, textvariable=temperature, width=6)
speedEntry.place(relx=.35, rely=0.6)
speedEntry.insert(0, 0.0)

#Combobox
#choose = ('Shear rate', 'Shear stress', 'Speed', 'Torque')

#reologyList = ttk.Combobox(predictLabel, width=10)
#reologyList.place(relx=.0, rely=0.6)
#reologyList['values'] = choose
#reologyList.current()
#reologyList.bind("<<ComboboxSelected>>", lambda _ : printr(reologyList.current()))

#reoEntry = tk.Entry(predictLabel, textvariable=reo_var, width=5)
#reoEntry.place(relx=.35, rely=0.6)
#reoEntry.insert(0, 0.0)

predictButton = tk.Button(predictLabel, text="\tPredict\t", command=lambda: predict())
predictButton.place(relx=.4, rely=0.7)

#Table
frameTable = tk.LabelFrame(root, text="Dataset")
frameTable.place(height=396, width=450, relx=0.355, rely=0.2)

treeView = ttk.Treeview(frameTable)
treeView.place(relheight=1, relwidth=1)

#Scroll
treeScrollX = tk.Scrollbar(frameTable, orient='horizontal', command=treeView.xview)
treeScrollY = tk.Scrollbar(frameTable, orient='vertical', command=treeView.yview)
treeView.configure(xscrollcommand=treeScrollX.set, yscrollcommand=treeScrollY.set)
treeScrollX.pack(side="bottom", fill="x")
treeScrollY.pack(side="right", fill="y")

#Data
frameImport = tk.LabelFrame(root, text="Import data with last column as target")
frameImport.place(height=100, width=225, relx=0.355, rely=0)

#Buttons
buttonSearch = tk.Button(frameImport, text="Browse a file", command=lambda: search_file())
buttonSearch.place(relx=0.1, rely=0.4)
buttonLoad = tk.Button(frameImport, text="Load data", command=lambda: load_data())
buttonLoad.place(relx=0.5, rely=0.4)

#Label
labelFile = ttk.Label(frameImport, text="No File Selected")
labelFile.place(relx=0, rely=0)

#Fit
buildTree = tk.LabelFrame(root, text="Build tree")
buildTree.place(height=100, width=225, rely=0, relx=0.675)

splitLabel = tk.Label(buildTree, text="test size")
splitLabel.place(relx=0, rely=0)

testSizeEntry = tk.Spinbox(buildTree, from_=0.1, to= 0.5, textvariable=test_size, values =(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),wrap = False, width=6)
testSizeEntry.place(relx=0.22, rely=0.01)

buttonFit = tk.Button(buildTree, text="\tFit\t", command=lambda: fit())
buttonFit.place(relx=.3, rely=0.4)

root.mainloop()