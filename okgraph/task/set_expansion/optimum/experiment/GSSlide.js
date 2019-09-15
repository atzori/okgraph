/**
 * Copyright Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// [START apps_script_slides_progress]
/**
 * @OnlyCurrentDoc Adds progress bars to a presentation.
 */
var BAR_ID = 'PROGRESS_BAR_ID';
var BAR_HEIGHT = 5; // px
var PADDING = 5;
var TITLES_HEIGHT = 20;
var presentation = SlidesApp.getActivePresentation();

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
    "title": "Contribution",
  },{ 
    "title": "Project OKgraph",
  },{ 
    "title": "Optimized solution",
  },{ 
    "title": "Final considerations",
  }];

  var titleWidth = (presentation.getPageWidth() - PADDING * 2) / titles.length - PADDING ;
  var slides = presentation.getSlides();
  for (var i = 0; i < slides.length; ++i) {
    addTitles(slides[i], titleWidth, titles)
  }
}

function addTitles(slide, titleWidth, titles) {
    for (var titleIndex = 0; titleIndex < titles.length; ++titleIndex) {
        var x = titleWidth * titleIndex;
        var y = 0;
        var titleShape = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, 
            x + PADDING * (titleIndex+1), 
            y + PADDING, titleWidth, TITLES_HEIGHT)    
        //titleShape.getBorder().setTransparent();
        titleShape.getFill().setSolidFill("#D1D5D8", 1)
        titleShape.setLinkUrl(BAR_ID); // to allow remove it
        titleShape.getText().setText(titles[titleIndex].title)
        titleShape.getText().getTextStyle().setFontSize(12)
        titleShape.getText().getTextStyle().setBold(titles[titleIndex].bold || false)
        titleShape.getText().getTextStyle().setForegroundColor("#FFFFFF")
        titleShape.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER)
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