var brain = require('brain');
var sys = require('sys');
var stdin = process.openStdin();

/*
    Neural Network Sorting Numbers
    by Kory Becker 2015 http://primaryobjects.com/kory-becker

    This program trains a neural network to sort a set of 3-digit numbers. Did you know a neural network can sort? Neither did I, until I wrote this program!

    Neural Network Setup:
    The 3-digit numbers are fed into the neural network by separating each digit as an input.
    Digits are normalized to fall between 0-1 by dividing each digit by 9.
    The number of inputs are equal to the number of digits in the numbers being sorted. For 2 3-digit numbers, there would be 6 inputs.
    The number of outputs are equal to the number of inputs. The output cooresponds to the actual digits, in sorted form. Numbers in. Sorted numbers out.

    Results:
    A neural network with two hidden layers of 25 neurons each (not including the output layer), trained for 6,000 iterations at a learning rate of 0.3,
    produces a training/test accuracy of 78%/74%.
*/

NeuralNetworkManager = {
    net: new brain.NeuralNetwork({ hiddenLayers: [25, 25] }),

    train: function(data, onComplete) {
        NeuralNetworkManager.run(data, onComplete, true);
    },

    run: function(data, onComplete, isTraining) {
        var result = [];

        // Normalize the data.
        var data = NeuralNetworkManager.normalize(data);

        if (isTraining) {
            // Train neural network.
            console.log('Training');
            NeuralNetworkManager.net.train(data, { errorThresh: 0.0001, iterations: 20000, learningRate: 0.3, log: true, logPeriod: 100 });
        }
        
        // Run the neural network against each row in the data and determine accuracy.
        var correct = 0;
        for (var i in data) {
            // Get the output vector from the neural network for this row.
            var output = NeuralNetworkManager.net.run(data[i].input);
            // Denormalize the output back into digit form.
            var actual = NeuralNetworkManager.denormalize(output);
            // Denormalize the expected output back into digit form.
            var expected = NeuralNetworkManager.denormalize(data[i].output);

            // If any digit fails to match the expected output, then the neural network failed on this row.
            var failed = false;
            for (var j=0; j<output.length; j++) {
                if (actual[j] != expected[j]) {
                    // A digit is different in actual from expected. The network failed this row.
                    failed = true;
                    break;
                }
            }

            if (!failed) {
                correct++;
            }

            // Save history.
            var resultItem = {};
            resultItem.input = data[i].input;
            resultItem.actual = actual;
            resultItem.expected = expected;
            resultItem.correct = !failed;

            result.push(resultItem);
        }
        
        console.log('Accuracy: ' + ((correct / data.length) * 100).toFixed(2) + '%');
        
        // Call callback.
        if (onComplete) {
            onComplete(result);
        }
    },

    denormalize: function(output) {
        // Denormalizes a vector by converting the neural network 0-1 decimal value back into its original 0-9 digit form. Example: [0.6, 0.55, 0.83] => [5, 5, 8].
        var result = [];

        // Go through each decimal value in the vector.
        for (var i=0; i<output.length; i++) {
            var value = output[i];
            var digit = 0;
            
            // To normalize, we divide the digit by 9, giving us a value between 0 and 1 with each digit in a fixed range. To denormalize, we check the range of the value and set the appropriate digit.
            if (value <= 0.1) digit = 0;
            else if (value <= 0.2) digit = 1;
            else if (value <= 0.3) digit = 2;
            else if (value <= 0.4) digit = 3;
            else if (value <= 0.5) digit = 4;
            else if (value <= 0.6) digit = 5;
            else if (value <= 0.7) digit = 6;
            else if (value <= 0.8) digit = 7;
            else if (value <= 0.9) digit = 8;
            else if (value <= 1) digit = 9;

            result.push(digit);
        }

        return result;
    },

    normalize: function(data) {
        // Normalizes a vector by converting digits of 0-9 to fall between 0-1. For example: [5, 5, 8] => [0.556, 0.556, 0.889].
        var result = [];

        // Go through each row in the data.
        for (var i=0; i<data.length; i++) {
            var row = {};
            row.input = [];
            row.output = [];

            // Go through each digit in the input numbers and divide by 9, producing a value between 0-1.
            for (var j=0; j<data[i].input.length; j++) {
                row.input.push(data[i].input[j] / 9);
            }

            // Go through each digit in the output numbers and divide by 9, producing a value between 0-1.
            for (var j=0; j<data[i].output.length; j++) {
                row.output.push(data[i].output[j] / 9);
            }
            
            result.push(row);
        }
        
        return result;
    },

    randomFixedInteger: function (length) {
        return Math.floor(Math.pow(10, length-1) + Math.random() * 9 * Math.pow(10, length-1));
    },

    generateFormatted: function(totalCount, countPerRow, digits) {
        return NeuralNetworkManager.format(NeuralNetworkManager.generate(totalCount, countPerRow, digits));
    },

    generate: function(totalCount, countPerRow, digits) {
        // Generates random numbers to use as training and test data.
        // Returns an array of rows with input and sorted output, in the form: [ { input: [ 987, 123 ], output: [ 123, 987 ] }, ... ]
        // totalCount: Total number of rows to generate in the set. countPerRow: Number of numbers to generate per row. For example, 3 will generate 3 numbers to sort. digits: Number of digits per number.
        var data = [];

        for (var i=0; i<totalCount; i++) {
            var row = {};

            // Generate n numbers.
            var numbers = [];
            for (var j=0; j<countPerRow; j++) {
                numbers.push(NeuralNetworkManager.randomFixedInteger(digits));
            }

            // Set input numbers and sorted numbers for output.
            row.input = numbers;
            row.output = numbers.slice(0).sort(function(a,b) { return a - b });

            data.push(row);
        }

        return data;
    },

    commaSeparateDigits: function(number) {
        var formatted = '';

        for (var i=0; i<number.length; i++) {
            if (formatted.length > 0) {
                formatted += ',';
            }

            formatted += number[i];
        }

        return formatted;
    },

    format: function(data) {
        // Converts generated data as follows:
        // [ { input: [ 987, 123 ], output: [ 123, 987 ] }, ... ]
        // to
        // [ { input: [ 9,8,7,1,2,3 ], output: [ 1,2,3,9,8,7 ] } ... ]
        var result = [];

        for (var i=0; i<data.length; i++) {
            var row = {};

            var formatted = '';
            for (var j=0; j<data[i].input.length; j++) {
                if (formatted.length > 0) {
                        formatted += ',';
                }
                formatted += NeuralNetworkManager.commaSeparateDigits(data[i].input[j].toString());
            }

            row.input = JSON.parse('[' + formatted + ']');

            formatted = '';
            for (var j=0; j<data[i].output.length; j++) {
                if (formatted.length > 0) {
                        formatted += ',';
                }
                formatted += NeuralNetworkManager.commaSeparateDigits(data[i].output[j].toString());
            }

            row.output = JSON.parse('[' + formatted + ']');

            result.push(row);
        }

        return result;
    },

    formatInput: function(line) {
        // Converts user input in the form 10,20,30 to [ { input: [ 10,20,30 ], output: [ 10,20,30 ] } ] and finally to [ { input: [ 1,0,2,0,3,0 ], output: [1,0,2,0,3,0 ] }].
        var row = {};

        row.input = JSON.parse('[' + line + ']');
        row.output = row.input.slice(0).sort(function(a,b) { return a - b });

        return NeuralNetworkManager.format([ row ]);
    }
};

/*
//
// Example generated data for 2 3-digit sorting.
// Each digit in the 3-digit number is separated as its own input into the neural network.
//
var trainingData = [
{input: [1,2,3,9,8,7], output: [1,2,3,9,8,7]},
{input: [3,2,4,1,5,6], output: [1,5,6,3,2,4]},
{input: [9,2,1,6,7,4], output: [6,7,4,9,2,1]},
{input: [5,1,7,6,9,8], output: [5,1,7,6,9,8]},
{input: [1,6,2,1,4,5], output: [1,4,5,1,6,2]},
{input: [8,9,1,6,0,5], output: [6,0,5,8,9,1]}
];

var testData = [
{input: [1,6,2,1,4,5], output: [1,4,5,1,6,2]},
{input: [8,9,1,6,0,5], output: [6,0,5,8,9,1]},
{input: [2,3,4,6,7,8], output: [2,3,4,6,7,8]},
{input: [5,7,6,2,1,1], output: [2,1,1,5,7,6]},
{input: [6,7,0,9,9,2], output: [6,7,0,9,9,2]},
{input: [8,6,7,6,5,0], output: [6,5,0,8,6,7]}
];
*/

// Generate training and test data.
var trainingData = NeuralNetworkManager.generateFormatted(2000, 2, 3);
var testData = NeuralNetworkManager.generateFormatted(5000, 2, 3);

// Train the neural network on the training set.
NeuralNetworkManager.train(trainingData, function(result) {
    // Test the neural network on the cross-validation set.
    NeuralNetworkManager.run(testData, function(result) {
        // Allow user input.
        console.log('Enter two 3-digit numbers (example 321,145):');

        stdin.addListener("data", function(line) {
            // note: line is an object, and when converted to a string it will end with a linefeed. So we (rather crudely) account for that with toString() and then substring().
            line = line.toString().substring(0, line.length-1);
            
            // Convert the input into a form readable by the neural network.
            var input = NeuralNetworkManager.formatInput(line);

            // Run the network on the input.
            NeuralNetworkManager.run(input, function(result) {
                var output = JSON.stringify(result[0].actual);
                console.log('Output: ' + output.substr(1, output.length - 2));
                console.log('');
                console.log(result);
            });
        });        
    });
});