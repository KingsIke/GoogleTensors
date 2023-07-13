import { TRAINING_DATA } from "http://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

const INPUTS = TRAINING_DATA.inputs

const OUTPUTS = TRAINING_DATA.outputs


tf.util.shuffleCombo(INPUTS, OUTPUTS)

const INPUTS_TENSOR = tf.tensor2d(INPUTS);

const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

function normalize(tensor, min, max) {
    const result = tf.tidy(function () {

        const MIN_VALUES = min || tf.min(tensor, 0);

        const MAX_VALUES = max || tf.max(tensor, 0);

        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES)

        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

        const NORMALIZE_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

        return { NORMALIZE_VALUES, MIN_VALUES, MAX_VALUES }
    })
    return result
}

const FEATURE_RESULTS = normalize(INPUTS_TENSOR)

console.log('Normalize Value:')
FEATURE_RESULTS.NORMALIZE_VALUES.print()



console.log('Min Value:')
FEATURE_RESULTS.MIN_VALUES.print()



console.log('Max Value:')
FEATURE_RESULTS.MAX_VALUES.print()

INPUTS_TENSOR.dispose()

stopped at 26..try console all those  values inside ...