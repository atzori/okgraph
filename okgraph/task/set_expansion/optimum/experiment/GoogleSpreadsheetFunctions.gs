Logger.clear()




function getWorstCase() {
    var filteredRows = getFilteredRows();
    Logger.log(filteredRows[ColumnIndex.objective_metric])
    return "Codice OK";
}

getWorstCase()





var ColumnIndex = function () {
    this.INFO = 7; // INFO: OPTIMIZED, CENTROID
    this.optim_algo = 8; // optim_algo: BFGS, nelder-mead, powell.....
    this.objective_metric = 9; // objective_metric: AP@k
    this.we_model = 36; // we_model: models/GoogleNews-vectors-negative300.magnitude
    this.ground_truth_name = 42; // ground_truth_name: usa_states
}

function getMaxInColumn(column) {
  var colArray = SpreadsheetApp.getActiveSheet().getDataRange().getValues()
  var maxInColumn = colArray.sort(function(a,b){return b-a})[0][0];
  SpreadsheetApp.getActiveSheet().getRange('I24').setValue('aaa');
  SpreadsheetApp.getActiveSheet().getRange('I24').setValue(maxInColumn);
}

function getSheet(name) {
  var sheets = SpreadsheetApp.getActiveSpreadsheet().getSheets();
  for (var i=0; i<sheets.length; i++) { 
    if (sheets[i].getName() === name) {
      return sheets[i];
    }
  }
  return null;
}

function getFilteredRows() {
  var sheetName = 'Results';
  var filteredRows = getSheet(sheetName).getRange("A1:AS8000").getValues()
  var sheetAnalysis = getSheet("AnalysisV2").getRange("A1:Z100");
  
  var valueOf_INFO = "OPTIMIZED"; // sheetAnalysis.getCell(5,5).getValue()
  var valueOf_optim_algo = sheetAnalysis.getCell(4,1).getValue();
  var valueOf_objective_metric = sheetAnalysis.getCell(4,3).getValue();
  var valueOf_we_model = sheetAnalysis.getCell(6,1).getValue();
  var valueOf_ground_truth_name = sheetAnalysis.getCell(7,1).getValue();

  filteredRows = filteredRows.filter(filterChoosen(valueOf_INFO, 
                                                   valueOf_optim_algo,
                                                   valueOf_objective_metric,
                                                   valueOf_we_model,
                                                   valueOf_ground_truth_name
                                                  ))
  return filteredRows;
}

var filterChoosen = function (
  valueOf_INFO,
  valueOf_optim_algo,
  valueOf_objective_metric,
  valueOf_we_model,
  valueOf_ground_truth_name
) {
    return function (row) {
      return row[ColumnIndex.INFO] === valueOf_INFO 
             && row[ColumnIndex.optim_algo] === valueOf_optim_algo
             && row[ColumnIndex.objective_metric] === valueOf_objective_metric
             && row[ColumnIndex.we_model] === valueOf_we_model
             && row[ColumnIndex.ground_truth_name] === valueOf_ground_truth_name
    }
}

