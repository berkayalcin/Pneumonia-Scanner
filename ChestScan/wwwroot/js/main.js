
$(document).ready(function () {
    //$('#results').fadeOut(1);
    initializeTensorModel();
});

function submitModel() {
    if (!window.model) {
        swal("Hata Oluştu!", "Model Oluşturulurken hata oluştu! Lütfen tekrar deneyin.", "error");
        return;
    }
    var image = document.getElementById('validatedCustomFile').value;
    if (!image) {
        swal("Hata Oluştu!", "Resim seçilirken hata oluştu.", "error");
    }
    console.warn(image);
    predict();
}

async function predict() {
    //tf.ENV.set('WEBGL_PACK', false)
    //const model = await tf.loadLayersModel('https://localhost:44329/model/model.json');
    tf.tidy(() => {
        var image = tf.browser.fromPixels(document.getElementById('image')).toFloat();
        const offset = tf.scalar(127.5);
        const normalized = image.sub(offset).div(offset);
        const batched = normalized.reshape([1, window.image_size, window.image_size, 3]);
        const prediction = window.model.predict(batched);
        const tensorData = prediction.dataSync();
        const normalProb = tensorData[0];
        const anormalProb = tensorData[1];
        $('#zatureLabel').html('Zatüre Olma İhtimali : ' + anormalProb.toString().substring(0,4));
        $('#normalLabel').html('Normal Olma İhtimali : ' + normalProb.toString().substring(0, 4));
        $('#results').removeAttr('hidden');
        $('#image').attr('hidden', 'hidden');
        console.warn(tensorData);

    });
}
async function initializeTensorModel() {
    tf.ENV.set('WEBGL_PACK', false)
    const model = await tf.loadLayersModel('https://localhost:44329/model/model.json');
    window.model = model;
    window.image_size = 224;
};

function onFileSelected(event) {
    $('#results').attr('hidden','hidden');
    $('#image').removeAttr('hidden');

    var selectedFile = event.target.files[0];
    var reader = new FileReader();
    var imgtag = document.getElementById("image");
    imgtag.title = selectedFile.name;
    reader.onload = function (event) {
        imgtag.src = event.target.result;
    };
    reader.readAsDataURL(selectedFile);

}