var BAR_ID = 'PROGRESS_BAR_ID';
var BAR_HEIGHT = 5; // px
var PADDING = 5;
var TITLES_HEIGHT = 20;
var presentation = SlidesApp.getActivePresentation();

var C = {
    transparent: "NO COLOR",
    black: "#000000",
    gray: "#D1D5D8",
    white: "#FFFFFF",

};

var UI = {
    // title bar
    backgTitleBar: C.black,
    backgEachTitle: C.transparent,
    backgSelectedTitle: C.gray,
    textColorSelected: C.black,
    textColorNotSelected: C.white,
    titleBarBorderColor: C.transparent,
    // slide number
    textSlideNumberColor: C.white,
};
/**
 * Runs when the add-on is installed.
 * @param {object} e The event parameter for a simple onInstall trigger. To
 *     determine which authorization mode (ScriptApp.AuthMode) the trigger is
 *     running in, inspect e.authMode. (In practice, onInstall triggers always
 *     run in AuthMode.FULL, but onOpen triggers may be AuthMode.LIMITED or
 *     AuthMode.NONE.)
 */
function onInstall(e) {
    onOpen();
}

/**
 * Trigger for opening a presentation.
 * @param {object} e The onOpen event.
 */
function onOpen(e) {
    SlidesApp.getUi().createAddonMenu()
        .addItem('Show progress bar', 'createBars')
        .addItem('Hide progress bar', 'deleteBars')
        .addToUi();
}

/**
 * Create a rectangle on every slide with different bar widths.
 */
function createBars() {
    deleteBars(); // Delete any existing progress bars
    var titles = [{
        title: "Intro",
        slide: {
            from: 0,
            to: 6,
        }
    }, {
        title: "Contribution",
        slide: {
            from: 7,
            to: 7,
        }
    }, {
        title: "Project OKgraph",
        slide: {
            from: 8,
            to: 13,
        }
    }, {
        title: "Optimized solution",
        slide: {
            from: 14,
            to: 31,
        }
    }, {
        title: "Final considerations",
        slide: {
            from: 32,
            to: 32,
        }
    }];

    var titleWidth = (presentation.getPageWidth() - PADDING * 2) / titles.length - PADDING;
    var slides = presentation.getSlides();
    for (var i = 1; i < slides.length - 1; ++i) {
        addTitlesBackground(slides[i]);
        addTitles(slides[i], titleWidth, titles, i);
        addSlidesLeftNumberTriangle(slides[i]);
        addSlidesLeftNumber(slides[i], i + 1, slides.length)
    }
}

function addSlidesLeftNumberTriangle(slide) {
    var slideWidth = presentation.getPageWidth();
    var slideHeight = presentation.getPageHeight();
    var w = 60;
    var h = 30;
    var scaleFactor = 1.5;

    var backgNumbShape = slide.insertShape(SlidesApp.ShapeType.TRIANGLE,
        slideWidth - w * scaleFactor, slideHeight - h * scaleFactor, w * scaleFactor, h * scaleFactor);
    backgNumbShape.setRotation(135)
        .setLeft(slideWidth - 40 * scaleFactor)
        .setTop(slideHeight - 24 * scaleFactor);

    backgNumbShape.setLinkUrl(BAR_ID); // to allow remove it
    if (UI.backgTitleBar !== C.transparent) {
        backgNumbShape.getFill().setSolidFill(UI.backgTitleBar, 1)
    }
    if (UI.titleBarBorderColor === C.transparent) {
        backgNumbShape.getBorder().setTransparent();
    } else {
        backgNumbShape.getBorder().setSolidFill(UI.titleBarBorderColor)
    }
}

function addSlidesLeftNumber(slide, actualSlideNum, totalSlide) {
    var slideWidth = presentation.getPageWidth();
    var slideHeight = presentation.getPageHeight();
    var w = 60;
    var h = 30;

    var titleShape = slide.insertTextBox(actualSlideNum + '/' + totalSlide, slideWidth - w, slideHeight - h, w, h);
    setVerticalAlignment(DocumentApp.VerticalAlignment.BOTTOM);
    titleShape.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.END);
    titleShape.getText().getTextStyle().setForegroundColor(UI.textSlideNumberColor);
    titleShape.getText().getTextStyle().setFontSize(10);
    titleShape.setLinkUrl(BAR_ID); // to allow remove it
}

function addTitlesBackground(slide) {
    var slideWidth = presentation.getPageWidth();
    var titleShape = slide.insertShape(SlidesApp.ShapeType.RECTANGLE,
        0, 0, slideWidth, TITLES_HEIGHT + PADDING * 2);
    titleShape.setLinkUrl(BAR_ID); // to allow remove it
    if (UI.backgTitleBar !== C.transparent) {
        titleShape.getFill().setSolidFill(UI.backgTitleBar, 1)
    }
    if (UI.titleBarBorderColor === C.transparent) {
        titleShape.getBorder().setTransparent();
    } else {
        titleShape.getBorder().setSolidFill(UI.titleBarBorderColor)
    }
}

function addTitles(slide, titleWidth, titles, actualSlideIndex) {
    for (var titleIndex = 0; titleIndex < titles.length; ++titleIndex) {
        var isSelected = actualSlideIndex >= titles[titleIndex].slide.from - 1 &&
            actualSlideIndex <= titles[titleIndex].slide.to - 1;
        var x = titleWidth * titleIndex;
        var y = 0;
        var titleShape = slide.insertShape(SlidesApp.ShapeType.RECTANGLE,
            x + PADDING * (titleIndex + 1),
            y + PADDING, titleWidth, TITLES_HEIGHT);
        titleShape.setLinkUrl(BAR_ID); // to allow remove it
        //titleShape.getBorder().setTransparent();

        // background color
        var alpha = UI.backgEachTitle == C.transparent ? 0 : 1;
        var color = UI.backgEachTitle == C.transparent ? "#000000" : UI.backgEachTitle;
        if (isSelected) {
            alpha = UI.backgSelectedTitle == C.transparent ? 0 : 1;
            color = UI.backgSelectedTitle == C.transparent ? "#000000" : UI.backgSelectedTitle;
        }
        titleShape.getFill().setSolidFill(color, alpha);

        titleShape.getText().setText(titles[titleIndex].title);
        titleShape.getText().getTextStyle().setFontSize(12);
        titleShape.getText().getTextStyle().setBold(titles[titleIndex].bold || false);

        // Text color
        var textColor = isSelected ? UI.textColorSelected : UI.textColorNotSelected;
        titleShape.getText().getTextStyle().setForegroundColor(textColor);
        titleShape.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);
    }
}


/**
 * Deletes all progress bar rectangles.
 */
function deleteBars() {
    var slides = presentation.getSlides();
    for (var i = 0; i < slides.length; ++i) {
        var elements = slides[i].getPageElements();
        for (var j = 0; j < elements.length; ++j) {
            var el = elements[j];
            if (el.getPageElementType() === SlidesApp.PageElementType.SHAPE &&
                el.asShape().getLink() &&
                el.asShape().getLink().getUrl() === BAR_ID) {
                el.remove();
            }
        }
    }
}
// [END apps_script_slides_progress]