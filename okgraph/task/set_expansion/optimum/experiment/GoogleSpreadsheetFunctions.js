Logger.clear();

var ColumnIndex = function () { };
ColumnIndex.INFO = 7; // OPTIMIZED, CENTROID
ColumnIndex.optim_algo = 8; // BFGS, nelder-mead, powell.....
ColumnIndex.objective_metric = 9; // AP@k
ColumnIndex.objective_metric_result = 10; // 0.348842 ...
ColumnIndex.Pa50 = 14; // 0.98 ...
ColumnIndex.optim_message = 23; // Success..
ColumnIndex.we_model = 36; //  models/GoogleNews-vectors-negative300.magnitude
ColumnIndex.ground_truth_name = 42; // usa_states
ColumnIndex.experimentId = 43;

var MySettings = function () { };
MySettings.sheetResultsName = "Results";
MySettings.resultsRange = "A1:AS5200";


var theCase = {succ: true, imprV: 2 }; // W&BC(AP@k)
//var theCase = {succ: true, imprV: 1}  // W&BC(IMPR)
//var theCase = {succ: false, imprV: 2} // W&BCAll(AP@k)
//var theCase = {succ: false, imprV: 1} // W&BCAll(IMPR)



MySettings.only_successful_enabled = theCase.succ;
MySettings.improvementV = theCase.imprV;
if (MySettings.improvementV == 2) {
    MySettings.sheetWorstAndBestCasesName = MySettings.only_successful_enabled ? "W&BC(AP@k)" : "W&BCAll(AP@k)";
} else {
    MySettings.sheetWorstAndBestCasesName = MySettings.only_successful_enabled ? "W&BC(IMPR)" : "W&BCAll(IMPR)";
}



var sheetWorstAndBestCases = getSheet(MySettings.sheetWorstAndBestCasesName);
var sheetResults = getSheet(MySettings.sheetResultsName);

function searchWorstAndBestCases() {

    var objective_metric_list = ["AP@k"];
    var optim_algo_list = ['powell', 'nelder-mead', 'BFGS', 'Newton-CG', 'CG', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'COBYLA'];
    // var optim_algo_list = ['nelder-mead', 'BFGS', 'Newton-CG', 'CG', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'COBYLA'];
    var we_model_list = ["models/GoogleNews-vectors-negative300.magnitude", "models/glove.840B.300d.magnitude"];
    var ground_truth_name_list = ["usa_states", "universe_solar_planets", "kings_of_rome", "king_of_rome", "periodic_table_of_elements", "period_7_element"];

    var row_i = 3;
    var total = [objective_metric_list, optim_algo_list, we_model_list, ground_truth_name_list]
        .reduce(function (prev, curr) {
            return ((prev && prev.length) || prev) * curr.length
        });
    Logger.info("total: " + total);
    sheetWorstAndBestCases.getRange("A1").setValue("Wait.. -" + total);
    var allRows = sheetResults.getRange(MySettings.resultsRange);
    Logger.info("allRows: " + allRows.length);

    optim_algo_list.forEach(function (optim_algo) {
        var filters1 = {
            optim_algo: optim_algo,
        };
        var preFilteredRows = getFilteredRowsByFilters(allRows, filters1);
        Logger.info("preFilteredRows: " + preFilteredRows.length);

        objective_metric_list.forEach(function (objective_metric) {
            we_model_list.forEach(function (we_model) {
                ground_truth_name_list.forEach(function (ground_truth_name) {
                    var filters2 = {
                        objective_metric: objective_metric,
                        optim_algo: optim_algo,
                        we_model: we_model,
                        ground_truth_name: ground_truth_name,
                    };
                    var rwawabc = addRowWithAverageWorstAndBestCase(preFilteredRows, sheetWorstAndBestCases, filters2, row_i);
                    if (rwawabc) { // if something has been added...
                        row_i = row_i + 1;
                    }
                    total = total - 1;
                    sheetWorstAndBestCases.getRange("A1").setValue("Wait.. -" + total + " [" + row_i + "]");
                    deleteRow(sheetWorstAndBestCases, row_i)
                });
            });
        });
    });
}

var cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V'];
function addRowWithAverageWorstAndBestCase(rows, sheet, filters, row_i) {

    var col_i = 0;
    var filteredRows = getFilteredRowsByFilters(rows, filters);
    // Logger.log('filters: ' + JSON.stringify(filters, null, 2))
    Logger.log("filteredRows.length: " + filteredRows.length);
    if (filteredRows.length <= 0) { return false; }

    var bestAndWorstCases = getBestAndWorstCaseFrom(filteredRows);
    var rowsOfWorstCaseExperiment = MySettings.improvementV === 1 ? bestAndWorstCases.rowsOfWorstCaseExperiment : bestAndWorstCases.rowsOfWorstCaseExperimentV2;
    var rowsOfBestCaseExperiment = MySettings.improvementV === 1 ? bestAndWorstCases.rowsOfBestCaseExperiment : bestAndWorstCases.rowsOfBestCaseExperimentV2;
    var averageCaseObj = MySettings.improvementV === 1 ? bestAndWorstCases.averageCaseObj : bestAndWorstCases.averageCaseObjV2;
    if (!rowsOfWorstCaseExperiment || !rowsOfBestCaseExperiment) { return false; }

    var worstCaseExperimentId = rowsOfWorstCaseExperiment.centroid[ColumnIndex.experimentId];
    var bestCaseExperimentId = rowsOfBestCaseExperiment.centroid[ColumnIndex.experimentId];
    if (!worstCaseExperimentId || !bestCaseExperimentId) {
        Logger.log("Not found: worstCaseExperimentId: " + worstCaseExperimentId);
        Logger.log("Not found: bestCaseExperimentId: " + bestCaseExperimentId);
        return false;
    }

    var worstCaseObj = getDataObjOfRowsTwin(rowsOfWorstCaseExperiment);
    var bestCaseObj = getDataObjOfRowsTwin(rowsOfBestCaseExperiment);

    var today = new Date();
    var time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
    var dateTime = time;

    sheet.getRange(cols[col_i++] + row_i).setValue(dateTime);
    sheet.getRange(cols[col_i++] + row_i).setValue(bestAndWorstCases.numberOfUsedRows);
    sheet.getRange(cols[col_i++] + row_i).setValue(filters.optim_algo);

    // worst case
    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.centroidResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.optimizedResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.improvement);

    // average case
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.centroidResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.optimizedResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.improvementFromAVG);
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.improvementFromList);

    // best case
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.centroidResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.optimizedResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.improvement);


    sheet.getRange(cols[col_i++] + row_i).setValue(filters.ground_truth_name);
    sheet.getRange(cols[col_i++] + row_i).setValue(getWEModelLabel(filters.we_model));
    sheet.getRange(cols[col_i++] + row_i).setValue(filters.objective_metric);


    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.experimentId);
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.experimentId);
    return true;

}

function getWEModelLabel(model) {
    const models = {
        'models/GoogleNews-vectors-negative300.magnitude': 'Google News 100B (W2V)',
        'models/glove.6B.300d.magnitude': 'Wikipedia 2014 + Gigaword 5 6B (GloVe)',
        'models/glove-lemmatized.6B.300d.magnitude': 'Wikipedia 2014 + Gigaword 5 6B lemmatized (GloVe)',
        'models/glove.840B.300d.magnitude': 'Common Crawl 840B (GloVe)',
        'models/glove.twitter.27B.200d.magnitude': 'Twitter 27B (GloVe)',
        'models/wiki-news-300d-1M.magnitude': 'English Wikipedia 2017 16B (fastText)',
        'models/wiki-news-300d-1M-subword.magnitude': 'English Wikipedia 2017 + subword 16B (fastText)',
        'models/crawl-300d-2M.magnitude': 'Common Crawl 600B (fastText)',
    };
    return models[model] || model;
}

function getImprovement(centroidResult, optimizedResult) {
    return centroidResult === 0 ? 0 : (optimizedResult / centroidResult) - 1;
}

function getDataObjOfRowsTwin(rowsObject) {
    var rowCentroid = rowsObject.centroid;
    var rowOptimized = rowsObject.optimized;
    var experimentId = rowCentroid[ColumnIndex.experimentId];
    var centroidResult = rowCentroid[ColumnIndex.objective_metric_result];
    var optimizedResult = rowOptimized[ColumnIndex.objective_metric_result];
    return {
        centroidResult: centroidResult,
        optimizedResult: optimizedResult,
        improvement: getImprovement(centroidResult, optimizedResult),
        experimentId: experimentId,
    }
}

// function onEdit(e) {
//     // e = {authMode=LIMITED, range=Range, source=Spreadsheet, oldValue=nelder-mead, user=emanuelemameli@gmail.com, value=BFGS}
//     var col = e.range.getColumn()
//     var row = e.range.getRow()
//     if (col === 1) {
//         if ([4, 6, 7].indexOf(row) > -1) {
//             // use case has changed
//             searchWorstAndBestCases()
//         }
//     }
// }



var filterChoosen = function (filters) {
    return function (row) {
        return (!filters.optim_algo || row[ColumnIndex.optim_algo] === filters.optim_algo)
            && (!filters.objective_metric || row[ColumnIndex.objective_metric] === filters.objective_metric)
            && (!filters.we_model || row[ColumnIndex.we_model] === filters.we_model)
            && (!filters.ground_truth_name || row[ColumnIndex.ground_truth_name] === filters.ground_truth_name)
    }
};


function getRowByExperimentId(rows, experimentId) {
    var rowsExperiment = rows.filter(function (row) {
        return row[ColumnIndex.experimentId] === experimentId;
    });
    if (rowsExperiment.length !== 2) {
        Logger.log("WARNING: it was expected to have 2 rows (experimentId: " + experimentId + ") but got " + rowsExperiment.length + ".")
    }
    return { // rowsObject
        centroid: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "CENTROID";
        })[0],
        optimized: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "OPTIMIZED";
        })[0],
    }
}

function isSuccessful(row) {
    var optim_message = row[ColumnIndex.optim_message];
    return optim_message === 'Optimization terminated successfully.'
}

function getBestAndWorstCaseFrom(filteredRows) {

    var worstCaseCentroidValue = 999;
    var worstCaseValue = 999;
    var bestCaseValue = -1;
    var worstCaseValueV2 = 999;
    var bestCaseValueV2 = -1;
    var rowsOfWorstCaseExperiment = null;
    var rowsOfBestCaseExperiment = null;
    var rowsOfWorstCaseExperimentV2 = null;
    var rowsOfBestCaseExperimentV2 = null;
    var experimentIds = [];
    var averageCaseObj = {
        centroidResult: -1,
        optimizedResult: -1,
        improvementFromList: -1,
        improvementFromAVG: -1,
    };
    var averageCaseObjV2 = {
        centroidResult: -1,
        optimizedResult: -1,
        improvementFromList: -1,
        improvementFromAVG: -1,
    };
    var centroidList = [];
    var optimizedList = [];
    var improvementList = [];

    var centroidListV2 = [];
    var optimizedListV2 = [];
    var improvementListV2 = [];

    var numberOfUsedRows = 0;

    for (var i = 0; i < filteredRows.length; i++) {
        var row = filteredRows[i];
        var experimentId = row[ColumnIndex.experimentId];
        var isOptimizedOne = row[ColumnIndex.INFO] === "OPTIMIZED";
        if (isOptimizedOne && experimentIds.indexOf(experimentId) === -1
            && MySettings.only_successful_enabled === isSuccessful(row)
        ) {
            experimentIds.push(experimentId);
            var experimentRows = getRowByExperimentId(filteredRows, experimentId);
            if (experimentRows.centroid && experimentRows.optimized) {
                // prendi l'id della riga e recupera anche la riga del CENTROID
                var rowCentroid = experimentRows.centroid;
                var rowOptimized = experimentRows.optimized;

                // leggi objective_metric_result OPTIMIZED e sottrai a objective_metric_result CENTROID
                var objectiveMetricResultCentroid = rowCentroid[ColumnIndex.objective_metric_result];
                var objectiveMetricResultOptimized = rowOptimized[ColumnIndex.objective_metric_result];
                var actualDifference = objectiveMetricResultOptimized - objectiveMetricResultCentroid;

                // memorizza quella con l'improvement minore
                if (!rowsOfWorstCaseExperiment
                    || actualDifference < worstCaseValue
                    // objectiveMetricResultCentroid < worstCaseCentroidValue
                ) {
                    rowsOfWorstCaseExperiment = getACopy(experimentRows);
                    worstCaseValue = actualDifference;
                    worstCaseCentroidValue = objectiveMetricResultCentroid;
                }
                // memorizza quella con l'improvement maggiore
                if (!rowsOfBestCaseExperiment
                    || actualDifference >= bestCaseValue) {
                    rowsOfBestCaseExperiment = getACopy(experimentRows);
                    bestCaseValue = actualDifference;
                }


                // memorizza quella con l'improvement minore in termini di AP@k
                if (!rowsOfWorstCaseExperimentV2
                    || objectiveMetricResultCentroid < worstCaseValueV2
                ) {
                    rowsOfWorstCaseExperimentV2 = getACopy(experimentRows);
                    worstCaseValueV2 = objectiveMetricResultCentroid;
                }
                // memorizza quella con l'improvement maggiore in termini di AP@k
                if (!rowsOfBestCaseExperimentV2
                    || objectiveMetricResultOptimized >= bestCaseValueV2) {
                    rowsOfBestCaseExperimentV2 = getACopy(experimentRows);
                    bestCaseValueV2 = objectiveMetricResultOptimized;
                }


                numberOfUsedRows++;
                centroidList.push(objectiveMetricResultCentroid);
                optimizedList.push(objectiveMetricResultOptimized);
                improvementList.push(getImprovement(objectiveMetricResultCentroid, objectiveMetricResultOptimized));

                centroidListV2.push(objectiveMetricResultCentroid);
                optimizedListV2.push(objectiveMetricResultOptimized);
                improvementListV2.push(getImprovement(objectiveMetricResultCentroid, objectiveMetricResultOptimized));

            } else {
                Logger.log("Error experimentId doesn't exists or doesn't has optimized and centroid rows: " + experimentId);
                // Logger.log("experimentRows.centroid: " + experimentRows.centroid);
                // Logger.log("experimentRows.optimized: " + experimentRows.optimized);
            }
        }
    }

    averageCaseObj.centroidResult = centroidList.length > 0 ? centroidList.reduce(function (prev, curr) { return prev + curr }) / centroidList.length : 0;
    averageCaseObj.optimizedResult = optimizedList.length > 0 ? optimizedList.reduce(function (prev, curr) { return prev + curr }) / optimizedList.length : 0;
    averageCaseObj.improvementFromList = improvementList.length > 0 ? improvementList.reduce(function (prev, curr) { return prev + curr }) / improvementList.length : 0;
    averageCaseObj.improvementFromAVG = getImprovement(averageCaseObj.centroidResult, averageCaseObj.optimizedResult);

    averageCaseObjV2.centroidResult = centroidListV2.length > 0 ? centroidListV2.reduce(function (prev, curr) { return prev + curr }) / centroidListV2.length : 0;
    averageCaseObjV2.optimizedResult = optimizedListV2.length > 0 ? optimizedListV2.reduce(function (prev, curr) { return prev + curr }) / optimizedListV2.length : 0;
    averageCaseObjV2.improvementFromList = improvementListV2.length > 0 ? improvementListV2.reduce(function (prev, curr) { return prev + curr }) / improvementListV2.length : 0;
    averageCaseObjV2.improvementFromAVG = getImprovement(averageCaseObjV2.centroidResult, averageCaseObjV2.optimizedResult);

    return {
        rowsOfWorstCaseExperiment: rowsOfWorstCaseExperiment,
        rowsOfBestCaseExperiment: rowsOfBestCaseExperiment,
        rowsOfWorstCaseExperimentV2: rowsOfWorstCaseExperimentV2,
        rowsOfBestCaseExperimentV2: rowsOfBestCaseExperimentV2,
        averageCaseObj: averageCaseObj,
        averageCaseObjV2: averageCaseObjV2,
        numberOfUsedRows: numberOfUsedRows,
    };
}


function getACopy(par) {
    return JSON.parse(JSON.stringify(par))
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

function getFilteredRowsByFilters(rows, filters) {
    var filteredRows = Array.isArray(rows) ? rows : rows.getValues();
    filteredRows = filteredRows.filter(filterChoosen(filters));
    return filteredRows;
}

function deleteRow(sheet, row) {
    cols.forEach(function (col) {
        sheet.getRange(col + row).setValue('');
    })
}

