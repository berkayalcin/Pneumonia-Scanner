
$(document).ready(function () {
    start();
});
async function start() {
    tf.ENV.set('WEBGL_PACK', false)
    const model = await tf.loadLayersModel('https://localhost:44329/model/model.json');
    console.log(model);
    tf.tidy(() => {
        const IMAGE_SIZE = 224;

        var image = tf.browser.fromPixels(document.getElementById('image1')).toFloat();
        const offset = tf.scalar(127.5);
        const normalized = image.sub(offset).div(offset);
        const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        const prediction = model.predict(batched);
        const tensorData = prediction.dataSync();
        const normalProb = tensorData[0];
        const anormalProb = tensorData[1];
        if (normalProb < 0.5) {
            $('#resultLabel').addClass("text-danger");
            $('#resultLabel').html('Zature Belirtisi.');
        } else {
            $('#resultLabel').addClass("text-success");
            $('#resultLabel').html('Zature Belirtisi Bulunmadı.');
        }
        $('#resultLabel').removeAttr('hidden');
        $('#resultLabel').removeProp('hidden');
        console.log(tensorData);
    });
   
    //console.log(prediction);
}