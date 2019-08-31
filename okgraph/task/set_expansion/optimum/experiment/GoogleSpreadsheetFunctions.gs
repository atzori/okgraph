Logger.clear()


function getWorstCase() {
    var filteredRowsObj = getFilteredRows();
    var allRows = filteredRowsObj.allRows;
    var filteredRows = filteredRowsObj.filteredRows;
    Logger.log("filteredRows.length: " + filteredRows.length)

    var worstCaseExperimentId = getWorstCaseExperimentIdFrom(allRows, filteredRows)
    Logger.log("worstCaseExperimentId: " + worstCaseExperimentId)

    var rowsOfWorstCaseExperiment = getRowByExperimentId(allRows, worstCaseExperimentId);

    var rowCentroid = rowsOfWorstCaseExperiment["centroid"];
    var rowOptimized = rowsOfWorstCaseExperiment["optimized"];

    var today = new Date();
    var time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
    var dateTime = time;
    
    getSheet(MySettings.sheetAnalysisName).getRange('H25').setValue(dateTime);
    getSheet(MySettings.sheetAnalysisName).getRange('I25').setValue(rowCentroid[ColumnIndex.experimentId]);
    getSheet(MySettings.sheetAnalysisName).getRange('J25').setValue(rowCentroid[ColumnIndex.objective_metric_result]);
    getSheet(MySettings.sheetAnalysisName).getRange('K25').setValue(rowOptimized[ColumnIndex.objective_metric_result]);
}

function onEdit(e) {
    // e = {authMode=LIMITED, range=Range, source=Spreadsheet, oldValue=nelder-mead, user=emanuelemameli@gmail.com, value=BFGS}
    var col = e.range.getColumn()
    var row = e.range.getRow()
    if (col === 1) {
        if ([4, 6, 7].indexOf(row) > -1 ) {
            // use case has changed
            getWorstCase()
        }
    }
}

var ColumnIndex = function () { }
ColumnIndex.INFO = 7; // OPTIMIZED, CENTROID
ColumnIndex.optim_algo = 8; // BFGS, nelder-mead, powell.....
ColumnIndex.objective_metric = 9; // AP@k
ColumnIndex.objective_metric_result = 10; // 0.348842 ...
ColumnIndex.we_model = 36; //  models/GoogleNews-vectors-negative300.magnitude
ColumnIndex.ground_truth_name = 42; // usa_states
ColumnIndex.experimentId = 43;

var MySettings = function () { }
MySettings.sheetAnalysisName = "Worst&BestCases";
MySettings.sheetResultsName = "Results";
MySettings.resultsRange = "A1:AS1000";

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


function getRowByExperimentId(rows, experimentId) {
    var rowsExperiment = rows.filter(function (row) {
        return row[ColumnIndex.experimentId] === experimentId
    })
    return {
        centroid: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "CENTROID"
        })[0],
        optimized: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "OPTIMIZED"
        })[0],
    }
}


function getWorstCaseExperimentIdFrom(allRows, filteredRows) {
    var worstCaseExperimentId = null;
    var worstCaseValue = null;

    for (var i = 0; i < filteredRows.length; i++) {
        const rowOptimized = filteredRows[i];

        // prendi l'id della riga e recupera anche la riga del CENTROID
        var experimentId = rowOptimized[ColumnIndex.experimentId];
        rowCentroid = getRowByExperimentId(allRows, experimentId)["centroid"];

        // leggi objective_metric_result OPTIMIZED e sottrai a objective_metric_result CENTROID
        objectiveMetricResultOptimized = rowOptimized[ColumnIndex.objective_metric_result];
        objectiveMetricResultCentroid = rowCentroid[ColumnIndex.objective_metric_result];
        actualWorstCaseValue = objectiveMetricResultOptimized - objectiveMetricResultCentroid;

        // memorizza quella con l'improvement minore
        if (!worstCaseExperimentId || actualWorstCaseValue < worstCaseValue) {
            worstCaseExperimentId = experimentId;
            worstCaseValue = actualWorstCaseValue;
        }
    }
    return worstCaseExperimentId;
}




function getMaxInColumn(column) {
    var colArray = SpreadsheetApp.getActiveSheet().getDataRange().getValues()
    var maxInColumn = colArray.sort(function (a, b) { return b - a })[0][0];
    SpreadsheetApp.getActiveSheet().getRange('I24').setValue('aaa');
    SpreadsheetApp.getActiveSheet().getRange('I24').setValue(maxInColumn);
}

function getSheet(name) {
    var sheets = SpreadsheetApp.getActiveSpreadsheet().getSheets();
    for (var i = 0; i < sheets.length; i++) {
        if (sheets[i].getName() === name) {
            return sheets[i];
        }
    }
    return null;
}

function getFilteredRows() {
    var sheetName = MySettings.sheetResultsName;
    var allRows = getSheet(sheetName).getRange(MySettings.resultsRange).getValues();
    var filteredRows = getSheet(sheetName).getRange(MySettings.resultsRange).getValues()
    var sheetAnalysis = getSheet(MySettings.sheetAnalysisName).getRange("A1:Z100");

    var valueOf_INFO = "OPTIMIZED"; // sheetAnalysis.getCell(5,5).getValue()
    var valueOf_optim_algo = sheetAnalysis.getCell(4, 1).getValue();
    var valueOf_objective_metric = sheetAnalysis.getCell(4, 3).getValue();
    var valueOf_we_model = sheetAnalysis.getCell(6, 1).getValue();
    var valueOf_ground_truth_name = sheetAnalysis.getCell(7, 1).getValue();

    filteredRows = filteredRows.filter(filterChoosen(valueOf_INFO,
        valueOf_optim_algo,
        valueOf_objective_metric,
        valueOf_we_model,
        valueOf_ground_truth_name
    ))
    return {
        allRows: allRows,
        filteredRows: filteredRows
    };
}


