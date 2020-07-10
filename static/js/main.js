
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;

// this function gets called once the user drops the file onto the div
function handleFileSelect(evt) {
    evt.stopPropagation();
    evt.preventDefault();

    // Get the FileList object that contains the list of files that were dropped
    const files = evt.dataTransfer.files;

    // this UI is only built for a single file so just dump the first one
    file = files[0];
    const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
    loadAndViewImage(imageId);
}

function handleDragOver(evt) {
    evt.stopPropagation();
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
}

// Setup the dnd listeners.
const dropZone = document.getElementById('dicomImage');
dropZone.addEventListener('dragover', handleDragOver, false);
dropZone.addEventListener('drop', handleFileSelect, false);


cornerstoneWADOImageLoader.configure({
    beforeSend: function(xhr) {
        // Add custom headers here (e.g. auth tokens)
        //xhr.setRequestHeader('x-auth-token', 'my auth token');
    },
    useWebWorkers: true,
});

let loaded = false;

function loadAndViewImage(imageId,element) {
   
    const start = new Date().getTime();
    cornerstone.loadImage(imageId).then(function(image) {
        console.log(image);
        const viewport = cornerstone.getDefaultViewportForImage(element, image);
         cornerstone.displayImage(element, image, viewport);
        
/*
        function getTransferSyntax() {
            const value = image.data.string('x00020010');
            return value + ' [' + uids[value] + ']';
        }

        function getSopClass() {
            const value = image.data.string('x00080016');
            return value + ' [' + uids[value] + ']';
        }

        function getPixelRepresentation() {
            const value = image.data.uint16('x00280103');
            if(value === undefined) {
                return;
            }
            return value + (value === 0 ? ' (unsigned)' : ' (signed)');
        }

        function getPlanarConfiguration() {
            const value = image.data.uint16('x00280006');
            if(value === undefined) {
                return;
            }
            return value + (value === 0 ? ' (pixel)' : ' (plane)');
        }

        document.getElementById('transferSyntax').textContent = getTransferSyntax();
        document.getElementById('sopClass').textContent = getSopClass();
        document.getElementById('samplesPerPixel').textContent = image.data.uint16('x00280002');
        document.getElementById('photometricInterpretation').textContent = image.data.string('x00280004');
        document.getElementById('numberOfFrames').textContent = image.data.string('x00280008');
        document.getElementById('planarConfiguration').textContent = getPlanarConfiguration();
        document.getElementById('rows').textContent = image.data.uint16('x00280010');
        document.getElementById('columns').textContent = image.data.uint16('x00280011');
        document.getElementById('pixelSpacing').textContent = image.data.string('x00280030');
        document.getElementById('bitsAllocated').textContent = image.data.uint16('x00280100');
        document.getElementById('bitsStored').textContent = image.data.uint16('x00280101');
        document.getElementById('highBit').textContent = image.data.uint16('x00280102');
        document.getElementById('pixelRepresentation').textContent = getPixelRepresentation();
        document.getElementById('windowCenter').textContent = image.data.string('x00281050');
        document.getElementById('windowWidth').textContent = image.data.string('x00281051');
        document.getElementById('rescaleIntercept').textContent = image.data.string('x00281052');
        document.getElementById('rescaleSlope').textContent = image.data.string('x00281053');
        document.getElementById('basicOffsetTable').textContent = image.data.elements.x7fe00010 && image.data.elements.x7fe00010.basicOffsetTable ? image.data.elements.x7fe00010.basicOffsetTable.length : '';
        document.getElementById('fragments').textContent = image.data.elements.x7fe00010 && image.data.elements.x7fe00010.fragments ? image.data.elements.x7fe00010.fragments.length : '';
        document.getElementById('minStoredPixelValue').textContent = image.minPixelValue;
        document.getElementById('maxStoredPixelValue').textContent = image.maxPixelValue;
        */
     /*   const end = new Date().getTime();
        const time = end - start;
        document.getElementById('totalTime').textContent = time + "ms";
        document.getElementById('loadTime').textContent = image.loadTimeInMS + "ms";
        document.getElementById('decodeTime').textContent = image.decodeTimeInMS + "ms";
       */
    }, function(err) {
        alert(err);
    });
}

cornerstone.events.addEventListener('cornerstoneimageloadprogress', function(event) {
    const eventData = event.detail;
    const loadProgress = document.getElementById('loadProgress');
    loadProgress.textContent = `Image Load Progress: ${eventData.percentComplete}%`;
});

const element = document.getElementById('dicomImage');
cornerstone.enable(element);

/*document.getElementById('selectFile').addEventListener('change', function(e) {
    // Add the file to the cornerstoneFileImageLoader and get unique
    // number for that file
    const file = e.target.files[0];
    const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
    
});*/

$(document).ready(function () {
    function readURL(input) {
        if (input.files && input.files[0]) {
            const file = e.target.result;
            const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
            const element = document.getElementById('dicomImage');
            loadAndViewImage(imageId,element);
        }

    }
$("#selectFile").change(function (e) {
    $('.image-section').show();
    $('#btn-predict').show();
    $('#result1').text('');
    $('#result1').hide();
    $('#result2').text('');
    $('#result2').hide();
    $('#result3').text('');
    $('#result3').hide();
    const file = e.target.files[0];
    const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
    const element = document.getElementById('dicomImage');
    loadAndViewImage(imageId,element);
   // readURL(this);
});
 // Predict
 $('#btn-predict').click(function () {
    var form_data = new FormData($('#form')[0]);

    // Show loading animation
    $(this).hide();
    $('.loader').show();
    console.log("button clicked")
    // Make prediction by calling api /predict
    $.ajax({
        type: 'POST',
        url: '/predict',
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        async: true,
        success: function (data) {
            console.log("success");
            console.log(data);
            // Get and display the result
            $('.loader').hide();
            var image = new Image();
            // b64Data contains the string I have showed above from Java API
            //image.setAttribute('src', );
            //document.getElementById('userImage1').appendChild(image);
            $('#userImage1').attr('src','data:image/png;base64,' + data.maskrcnn);
            $('#userImage2').attr('src','data:image/png;base64,' + data.vgg);
            $('#userImage3').attr('src','data:image/png;base64,' + data.mobilenet);
            console.log(data.message);
            //$('#result1').css('background-image', 'url(' + data+ ')');
            $('#result1').hide();
            $('#result1').fadeIn(650);
            //$('#result1').val(data);
          /*  const file = data;
            const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
            const element1 = document.getElementById('result1');
            loadAndViewImage(imageId,element1);
            const element2 = document.getElementById('result2');
            loadAndViewImage(imageId,element2);
            const element3 = document.getElementById('result3');
            loadAndViewImage(imageId,element3);*/
            /*$('#result1').fadeIn(600);
            $('#result1').text(' Result:  ' + data);
            $('#result2').fadeIn(600);
            $('#result2').text(' Result:  ' + data);
            $('#result3').fadeIn(600);
            $('#result3').text(' Result:  ' + data);*/
            console.log('Success!');
        },
    });
});
});

