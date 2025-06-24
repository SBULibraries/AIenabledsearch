

(function () {
   "use strict";
   'use strict';

   var app = angular.module('viewCustom', ['angularLoad']);

   /* JQUERY */
   /* This code adds jQuery, which is required for some customizations */
   var jQueryScript = document.createElement("script");
   jQueryScript.src = "https://code.jquery.com/jquery-3.3.1.min.js";
   document.getElementsByTagName("head")[0].appendChild(jQueryScript);

  // Adds the chat button
  (function () {
    var lc = document.createElement('script'); lc.type = 'text/javascript'; lc.async = 'true';
    lc.src = ('https:' == document.location.protocol ? 'https://' : 'http://') + 'answers.library.stonybrook.edu/load_chat.php?hash=48e25e6ff5dac1a9254eb9e2064ddb35';
    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(lc, s);
  })();
  // End the chat button

/* REPORT A BROKEN LINK */
   // Report a problem link, provided by Tarrant County College, with modifications
   app.component('prmAlmaViewitAfter', {
      bindings: { parentCtrl: '<' },
      controller: 'prmAlmaViewitAfterController',
      templateUrl: 'custom/01SUNY_STB-01SUNY_STB/html/homepage/report_broken_link.html'
   });
   app.controller('prmAlmaViewitAfterController', [ function() {
      var vm = this;
      this.$onInit = function(){
            {
            vm.tcc_dialog_content = "<section id='report_a_broken_link_form'><h2>Report a Broken Link Form</h2>";
            vm.tcc_meta_inputs = "";
            try {
               /* add all data except abstract to the form in hidden fields */
               vm.tcc_info = vm.parentCtrl.item.pnx.addata;
               for(var property_name in vm.tcc_info) {
                  if(property_name!='abstract') {
                     vm.tcc_meta_inputs += "<input type='hidden' id='" + property_name + "' name='" + property_name + "' value='" + vm.tcc_info[property_name] + "'>";
                  }
               }

               vm.tcc_dialog_content += "<div class='report-metadata'>"; //start metadata box
               
               /* put the author data in the dialog (full name or last name/first name) */
               if(undefined!=vm.parentCtrl.item.pnx.addata.au) {
                  vm.tcc_dialog_content += vm.parentCtrl.item.pnx.addata.au + ". ";
               } else {
                  if(undefined!=vm.parentCtrl.item.pnx.addata.aulast) {
                     vm.tcc_dialog_content += vm.parentCtrl.item.pnx.addata.aulast + ", ";
                     if(undefined!=vm.parentCtrl.item.pnx.addata.aufirst) {
                        vm.tcc_dialog_content += vm.parentCtrl.item.pnx.addata.aufirst + ". ";
                     }
                  }
               }
               /* put the article title, the book title, or the journal title in the dialog (whichever comes first) */
               if(undefined!=vm.parentCtrl.item.pnx.addata.atitle) {
                  vm.tcc_dialog_content += "\"" + vm.parentCtrl.item.pnx.addata.atitle + ".\" ";
               } else if(undefined!=vm.parentCtrl.item.pnx.addata.btitle) {
                  vm.tcc_dialog_content += "\"" + vm.parentCtrl.item.pnx.addata.btitle + ".\" ";
               } else if(undefined!=vm.parentCtrl.item.pnx.addata.jtitle) {
                  vm.tcc_dialog_content += "\"" + vm.parentCtrl.item.pnx.addata.jtitle + ".\" ";
               }
               /* put the journal title in the dialog */
               if(undefined!=vm.parentCtrl.item.pnx.addata.jtitle) {
                  vm.tcc_dialog_content += "<em>" + vm.parentCtrl.item.pnx.addata.jtitle + "</em>, ";
               }
               /* put the volume in the dialog */
               if(undefined!=vm.parentCtrl.item.pnx.addata.volume) {
                  vm.tcc_dialog_content += "vol. " + vm.parentCtrl.item.pnx.addata.volume + ", ";
               }
               /* put the issue in the dialog */
               if(undefined!=vm.parentCtrl.item.pnx.addata.issue) {
                  vm.tcc_dialog_content += "no. " + vm.parentCtrl.item.pnx.addata.issue + ", ";
               }
               /* put the date in the dialog */
               if(undefined!=vm.parentCtrl.item.pnx.addata.date) {
                  vm.tcc_dialog_content += vm.parentCtrl.item.pnx.addata.date + ", ";
               }
               /* put the pages in the dialog */
               if(undefined!=vm.parentCtrl.item.pnx.addata.pages) {
                  vm.tcc_dialog_content += "pp. " + vm.parentCtrl.item.pnx.addata.pages + ".";
               } else if(undefined!=vm.parentCtrl.item.pnx.addata.spage) {
                  vm.tcc_dialog_content += "pp. " + vm.parentCtrl.item.pnx.addata.spage;
                  if(undefined!=vm.parentCtrl.item.pnx.addata.epage) {
                     vm.tcc_dialog_content += "-" + vm.parentCtrl.item.pnx.addata.epage;
                  }
                  vm.tcc_dialog_content += ".";
               }
               vm.tcc_dialog_content += "</div>"; //end metadata box class='report-metadata'
               
               // Open form question box
               vm.tcc_dialog_content += "<p><strong>An asterisk (<span class='req'>*</span>) indicates a required field.</strong></p>";
               // Begin Form
               vm.tcc_dialog_content += "<form id='tcc_bad_link_form' method='post'><div id='tcc_dialog_form_inputs' style='border:1px solid #eee;padding:.8rem;'>";
               // Email
               vm.tcc_dialog_content += "<div><label for='badlink_report_email'>Your SBU email address <span class='req'>*</span></label><p>Your email address may be used to notify you that the problem has been resolved.</p><input type='email' id='badlink_report_email' name='badlink_report_email' value='' required pattern='.*@.*'></div>";
               // Describe problem
               vm.tcc_dialog_content += "<div><label for='badlink_report_option_id'>Which of the following best describes the problem with this listing? <span class='req'>*</span></label><select name='badlink_report_option_id' id='badlink_report_option_id' class='' size='1'><option disabled selected value=''>-- select one --</option><option value='1' >The PDF is blank/missing pages</option><option value='2' >I received a page not found or proxy-related error</option><option value='3' >The link led to the resource's main page or prompted me to pay for access</option><option value='4' >The link went to another website not related to the desired resource</option><option value='5' >Full text was not available, only the abstract or citation</option><option value='6' >Other problem with the listing, or something else went wrong -- explain in the comments below</option></select></div>";
               // Which link?
               vm.tcc_dialog_content += "<div><label for='badlink_report_which_link'>If there are multiple links to the resource listed on the catalog page, which of the links is broken?</label><input type='text' id='badlink_report_which_link' name='badlink_report_which_link' value=''></div>";
               // Browser
               vm.tcc_dialog_content += "<div><label for='badlink_report_which_browser'>Which browser are you using? <span class='req'>*</span></label><select id='badlink_report_which_browser' name='badlink_report_which_browser' size='1'><option disabled selected value=''>-- select one --</option><option value='chrome'>Google Chrome</option><option value='ff'>Firefox</option><option value='safari'>Safari</option><option value='edge'>Microsoft Edge</option><option value='other'>Other (Please specify in comments below)</option></select></div>";
               // On or off campus
               vm.tcc_dialog_content += "<div><label for='badlink_report_on_off_campus'>Are you connecting from on or off campus? <span class='req'>*</span></label><select id='badlink_report_on_off_campus' name='badlink_report_on_off_campus' size='1'><option disabled selected value=''>-- select one --</option><option value='oncampus'>On campus</option><option value='offcampus'>Off campus</option></select></div>";
               // Comments
               vm.tcc_dialog_content += "<div><label for='badlink_report_comments'>Further information or comments</label><textarea id='badlink_report_comments' name='badlink_report_comments'></textarea></div>";
               // End input box
               vm.tcc_dialog_content += "</div>";
               // Add Metadata hidden inputs
               vm.tcc_dialog_content += vm.tcc_meta_inputs;
               vm.tcc_dialog_content += "<input class='tingle-btn tingle-btn--primary' type='submit' id='form_submit' value='Send Report' />";
               vm.tcc_dialog_content += "</form></section>";

               // Create and open the dialog
               tcc_modal.setContent(vm.tcc_dialog_content);


               // Removed jQuery validation code here

               // Direct submission logic
               $('#tcc_bad_link_form').on('submit', function(e) {
                  e.preventDefault();
              
                  // Validate email input
                  var email = $('#badlink_report_email').val();
                  if (!email || !email.match(/[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/)) {
                      alert('Please enter a valid email address.');
                      return;
                  }
              
                  var form_data = new FormData(this);
              
                  form_data.append('current_url', window.location.href);

                  var xhttp = new XMLHttpRequest();
                  xhttp.onreadystatechange = function() {
                      if (this.readyState === 4) {
                          console.log("Status:", this.status);
                          console.log("Response:", this.responseText);
                          if (this.status === 200) {
                              // Check PHP script's response
                              try {
                                  let response = JSON.parse(this.responseText);
                                  if (response.success) {
                                      alert('Email sent successfully!');
                                  } else {
                                      alert(response.message);
                                  }
                              } catch (error) {
                                  alert('Unexpected response: ' + this.responseText);
                              }
                          } else {
                              alert('Error in PHP response: ' + this.responseText);
                          }
                      }
                  };
                  xhttp.open("POST", "https://repo.api.library.stonybrook.edu/MATT/primo/betteremail.php", true);
                  xhttp.send(form_data);              
                  tcc_modal.close(); // Close modal after submission
              });           

            } catch(err){
               console.log(err);
            }
         }
      };
   }]);

   /* END REPORT A BROKEN LINK */
/* SEARCH AI TEST */ 
app.component('prmSearchBarAfter', {
   bindings: { parentCtrl: '<' },
   controller: 'prmSearchBarAfterController',
   templateUrl: 'custom/01SUNY_STB-Hartman/html/homepage/SEARCH_AI.html'
});

app.controller('prmSearchBarAfterController', [function() {
   var vm = this;

   this.$onInit = function() {
         try {
            // Initialize the dialog content
            vm.SEARCH_AI_dialog_content = "<section id='SEARCH_AI_form'><h2>SEARCH AI</h2>";
            vm.SEARCH_AI_meta_inputs = "";

            // Open form question box
            vm.SEARCH_AI_dialog_content += "<p><strong>Use this tool to enhance your search queries using AI!</strong></p>";
            vm.SEARCH_AI_dialog_content += "<p>Please type your research inquiry using natural language into the form below, and AI will formulate a search query in our catalog.</p>";
            vm.SEARCH_AI_dialog_content += "<p><i>Please note: this is a starting point for your research. For more advanced queries, you may need to make some adjustments. If AI is unable to expand your query, your original search query will be used instead.</i></p>";

            // Begin Form
            vm.SEARCH_AI_dialog_content += "<form class='search-label' id='searchForm' action='https://repo.api.library.stonybrook.edu/MATT/smartsearch/proxy.php' method='POST'>";
            vm.SEARCH_AI_dialog_content += "<div><input id='searchQuery' name='query' type='text' placeholder='Example: Books held by our library on Traumatic Brain Injury in adolescents.' size='50' required /></div>";
            vm.SEARCH_AI_dialog_content += "<input class='tingle-btn tingle-btn--primary' type='submit' id='form_submit' size='50' value='Search AI' />";
            vm.SEARCH_AI_dialog_content += "</form></section>";

            vm.SEARCH_AI_dialog_content += "<div id='SEARCH_AI_loading_overlay' style='display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(255,255,255,0.85);z-index:9999;text-align:center;'>";
            vm.SEARCH_AI_dialog_content +="<div style='position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);'>"
            vm.SEARCH_AI_dialog_content +="<div class='spinner'></div>"
            vm.SEARCH_AI_dialog_content +="<p style='font-size:18px;color:#333;margin-top:10px;'>Processing your request...</p>"
            vm.SEARCH_AI_dialog_content +="</div>"
            vm.SEARCH_AI_dialog_content +="</div>"
            ;

         // Create and open the dialog
         SEARCH_AI_modal.setContent(vm.SEARCH_AI_dialog_content);

         // Handle form submission
         $('#searchForm').on('submit', function (e) {
            e.preventDefault();
            const overlay = document.getElementById('SEARCH_AI_loading_overlay');
            overlay.style.display = 'block';

            const formData = new FormData(this);

            // Disable submit to avoid double submits
            const submitButton = document.getElementById('form_submit');
            submitButton.disabled = true;
            submitButton.value = 'Thinking...';

            fetch('https://repo.api.library.stonybrook.edu/MATT/smartsearch/proxy.php', {
               method: 'POST',
               body: formData,
            })
            .then(response => response.text())
            .then(url => {
               const isValidURL = /^https?:\/\/.+/.test(url.trim());
               if (!isValidURL) throw new Error('Invalid URL from backend');

               overlay.innerHTML = `
                  <div style="position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);text-align:center;">
                     <div style="font-size:48px; color:green;">✔️</div>
                     <p style="font-size:20px; color:#333; margin-top:10px;">Redirecting to your search results...</p>
                  </div>
               `;

               setTimeout(() => {
                  window.location.href = url.trim();
               }, 500);
            })
            .catch(error => {
               console.error('AI Search Error:', error);
               overlay.innerHTML = `
                  <div style="position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);text-align:center;">
                     <div style="font-size:48px; color:#990000;">⚠️</div>
                     <p style="font-size:18px; color:#333;">Something went wrong. Using original search input.</p>
                  </div>
               `;
               setTimeout(() => {
                  document.getElementById('searchForm').submit();
               }, 1200);
            })
            .finally(() => {
               // Keep spinner until redirect happens or fallback hits
               setTimeout(() => {
                  submitButton.disabled = false;
                  submitButton.value = 'Search AI';
               }, 4000); // If still stuck, allow resubmission after 4s
            });
         });

      } catch (err) {
         console.log(err);
      }
   };
}]);
      /* END  SEARCH AI TEST */

  /****************************************************************************************************/
  /*In case of CENTRAL_PACKAGE - comment out the below line to replace the other module definition*/
  /*var app = angular.module('centralCustom', ['angularLoad']);*/
  /****************************************************************************************************/

  var LOCAL_VID = "01SUNY_STB-Hartman";
  var EXT_SEARCH_NAME = "Other Search Options"
  
  angular.module('externalSearch', []).value('searchTargets', [{
     "name": "WorldCat",
     "url": "https://sbulibraries.worldcat.org/search?databaseList=&queryString=",
     "img": "./custom/"+ LOCAL_VID +"/img/WorldCat_Logo.png",
     mapping: function mapping(search) {
        if (Array.isArray(search)) {
           var ret = '';
           for (var i = 0; i < search.length; i++) {
              var terms = search[i].split(',');
              ret += ' ' + (terms[2] || '');
           }
           return ret;
        } else {
           var terms = search.split(',');
           return terms[2] || "";
        }
     }
  }, {
     "name": "Google Scholar",
     "url": "https://scholar.google.com/scholar?q=",
     "img": "./custom/"+ LOCAL_VID +"/img/Google-Logo.png",
     mapping: function mapping(search) {
        if (Array.isArray(search)) {
           var ret = '';
           for (var i = 0; i < search.length; i++) {
              var terms = search[i].split(',');
              ret += ' ' + (terms[2] || '');
           }
           return ret;
        } else {
           var terms = search.split(',');
           return terms[2] || "";
        }
     }
  }]).component('prmFacetAfter', {
     bindings: { parentCtrl: '<' },
     controller: ['externalSearchService', function (externalSearchService) {
        this.$onInit = function () {
           externalSearchService.controller = this.parentCtrl;
           externalSearchService.addExtSearch();
        };
     }]
  }).component('prmPageNavMenuAfter', {
     controller: ['externalSearchService', function (externalSearchService) {
        if (externalSearchService.controller) externalSearchService.addExtSearch();
     }]
  }).component('prmFacetExactAfter', {
     bindings: { parentCtrl: '<' },
     template: '\n      <div ng-if="name === \''+ EXT_SEARCH_NAME +'\'">\n          <div ng-hide="$ctrl.parentCtrl.facetGroup.facetGroupCollapsed">\n              <div class="section-content animate-max-height-variable">\n                  <div class="md-chips md-chips-wrap">\n                      <div ng-repeat="target in targets" aria-live="polite" class="md-chip animate-opacity-and-scale facet-element-marker-local4">\n                          <div class="md-chip-content layout-row" role="button" tabindex="0">\n                              <strong dir="auto" title="{{ target.name }}">\n                                  <a ng-href="{{ target.url + target.mapping(queries, filters) }}" target="_blank">\n                                      <img ng-src="{{ target.img }}" width="22" height="22" style="vertical-align:middle;"> {{ target.name }}\n                                  </a>\n                              </strong>\n                          </div>\n                      </div>\n                  </div>\n              </div>\n          </div>\n      </div>',
     controller: ['$scope', '$location', 'searchTargets', function ($scope, $location, searchTargets) {
        this.$onInit = function () {
           $scope.name = this.parentCtrl.facetGroup.name;
           $scope.targets = searchTargets;
           var query = $location.search().query;
           var filter = $location.search().pfilter;
           $scope.queries = Array.isArray(query) ? query : query ? [query] : false;
           $scope.filters = Array.isArray(filter) ? filter : filter ? [filter] : false;
        };
     }]
  }).factory('externalSearchService', function () {
     return {
        get controller() {
           return this.prmFacetCtrl || false;
        },
        set controller(controller) {
           this.prmFacetCtrl = controller;
        },
        addExtSearch: function addExtSearch() {
           var xx = this;
           var checkExist = setInterval(function () {
  
              if (xx.prmFacetCtrl.facetService.results[0] && xx.prmFacetCtrl.facetService.results[0].name != EXT_SEARCH_NAME) {
                 if (xx.prmFacetCtrl.facetService.results.name !== EXT_SEARCH_NAME) {
                    xx.prmFacetCtrl.facetService.results.unshift({
                       name: EXT_SEARCH_NAME,
                       displayedType: 'exact',
                       limitCount: 0,
                       facetGroupCollapsed: false,
                       values: undefined
                    });
                 }
                 clearInterval(checkExist);
              }
           }, 100);
        }
     };
  });
  /* ====== */
    
  //Load latest jquery
  app.component('prmTopBarBefore', {
      bindings: {parentCtrl: '<'},
      controller: function () {
        this.$onInit = function () {
          loadScript("//ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js", jquery_loaded);
        };
      },
      template: ''
    });

  // Begin BrowZine - Primo Integration...
  window.browzine = {
    api: "https://public-api.thirdiron.com/public/v1/libraries/94",
    apiKey: "698e9941-3f20-4fb2-92a0-317bc3a648ae",

    journalCoverImagesEnabled: true,

    journalBrowZineWebLinkTextEnabled: true,
    journalBrowZineWebLinkText: "Browse Journal",

    articleBrowZineWebLinkTextEnabled: true,
    articleBrowZineWebLinkText: "View Issue Contents",

    articlePDFDownloadLinkEnabled: true,
    articlePDFDownloadLinkText: "Download PDF",

    articleLinkEnabled: true,
    articleLinkText: "Read Article",

    printRecordsIntegrationEnabled: true,

    unpaywallEmailAddressKey: "LibraryERMS@stonybrook.edu",

    articlePDFDownloadViaUnpaywallEnabled: true,
    articlePDFDownloadViaUnpaywallText: "Download PDF (via Unpaywall)",

    articleLinkViaUnpaywallEnabled: true,
    articleLinkViaUnpaywallText: "Read Article (via Unpaywall)",

    articleAcceptedManuscriptPDFViaUnpaywallEnabled: true,
    articleAcceptedManuscriptPDFViaUnpaywallText: "Download PDF (Accepted Manuscript via Unpaywall)",

    articleAcceptedManuscriptArticleLinkViaUnpaywallEnabled: true,
    articleAcceptedManuscriptArticleLinkViaUnpaywallText: "Read Article (Accepted Manuscript via Unpaywall)",
  };

  browzine.script = document.createElement("script");
  browzine.script.src = "https://s3.amazonaws.com/browzine-adapters/primo/browzine-primo-adapter.js";
  document.head.appendChild(browzine.script);

  app.controller('prmSearchResultAvailabilityLineAfterController', function ($scope) {
    window.browzine.primo.searchResult($scope);
  });

  app.component('prmSearchResultAvailabilityLineAfter', {
    bindings: { parentCtrl: '<' },
    controller: 'prmSearchResultAvailabilityLineAfterController'
  });
  // ... End BrowZine - Primo Integration

  // FITZ - 20210123
  // FITZ - 20210629
  var jQueryScript1 = document.createElement("script");
  jQueryScript1.src = "https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js";
  document.getElementsByTagName("head")[0].appendChild(jQueryScript1);

  var jQueryScript2 = document.createElement("script");
  jQueryScript2.src = "https://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js";
  document.getElementsByTagName("head")[0].appendChild(jQueryScript2);

  var css1 = document.createElement("link");
  css1.src = "https://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/themes/smoothness/jquery-ui.css";
  css1.rel = "stylesheet";
  document.getElementsByTagName("head")[0].appendChild(css1);

  jQueryScript1.onload = function () {
    $(function () {
      //$.getScript("./custom/01SUNY_STB-01SUNY_STB/js/FitzScript.js");
      $.getScript("https://library.stonybrook.edu/ALMA/PayPal/primo.js");
    });
  };
})();

/* Keep report a broken link submit button enabled */
function checkForValidation() {
  // Direct submission now, so no need for validation check.
}
/* Tingle JavaScript plugin */
!function(t,o){"function"==typeof define&&define.amd?define(o):"object"==typeof exports?module.exports=o():t.tingle=o()}(this,function(){var t=!1;function o(t){this.opts=function(){for(var t=1;t<arguments.length;t++)for(var o in arguments[t])arguments[t].hasOwnProperty(o)&&(arguments[0][o]=arguments[t][o]);return arguments[0]}({},{onClose:null,onOpen:null,beforeOpen:null,beforeClose:null,stickyFooter:!1,footer:!1,cssClass:[],closeLabel:"Close",closeMethods:["overlay","button","escape"]},t),this.init()}function e(){this.modalBoxFooter&&(this.modalBoxFooter.style.width=this.modalBox.clientWidth+"px",this.modalBoxFooter.style.left=this.modalBox.offsetLeft+"px")}function s(t){-1!==this.opts.closeMethods.indexOf("escape")&&27===t.which&&this.isOpen()&&this.close()}function i(t){-1!==this.opts.closeMethods.indexOf("overlay")&&!function(t,o){for(;(t=t.parentElement)&&!t.classList.contains(o););return t}(t.target,"tingle-modal")&&t.clientX<this.modal.clientWidth&&this.close()}return o.prototype.init=function(){if(!this.modal)return function(){this.modal=document.createElement("div"),this.modal.classList.add("tingle-modal"),(0===this.opts.closeMethods.length||-1===this.opts.closeMethods.indexOf("overlay"))&&this.modal.classList.add("tingle-modal--noOverlayClose");this.modal.style.display="none",this.opts.cssClass.forEach(function(t){"string"==typeof t&&this.modal.classList.add(t)},this),-1!==this.opts.closeMethods.indexOf("button")&&(this.modalCloseBtn=document.createElement("button"),this.modalCloseBtn.type="button",this.modalCloseBtn.classList.add("tingle-modal__close"),this.modalCloseBtnIcon=document.createElement("span"),this.modalCloseBtnIcon.classList.add("tingle-modal__closeIcon"),this.modalCloseBtnIcon.innerHTML='<svg width="100%" height="100%" viewBox="0 0 24 24" y="240" xmlns="http://www.w3.org/2000/svg" fit="" preserveAspectRatio="xMidYMid meet" focusable="false"><path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" fill="#fff"></path></svg>',this.modalCloseBtnLabel=document.createElement("span"),this.modalCloseBtnLabel.classList.add("tingle-modal__closeLabel"),this.modalCloseBtnLabel.innerHTML=this.opts.closeLabel,this.modalCloseBtn.appendChild(this.modalCloseBtnIcon),this.modalCloseBtn.appendChild(this.modalCloseBtnLabel));this.modalBox=document.createElement("div"),this.modalBox.classList.add("tingle-modal-box"),this.modalBoxContent=document.createElement("div"),this.modalBoxContent.classList.add("tingle-modal-box__content"),this.modalBox.appendChild(this.modalBoxContent),-1!==this.opts.closeMethods.indexOf("button")&&this.modal.appendChild(this.modalCloseBtn);this.modal.appendChild(this.modalBox)}.call(this),function(){this._events={clickCloseBtn:this.close.bind(this),clickOverlay:i.bind(this),resize:this.checkOverflow.bind(this),keyboardNav:s.bind(this)},-1!==this.opts.closeMethods.indexOf("button")&&this.modalCloseBtn.addEventListener("click",this._events.clickCloseBtn);this.modal.addEventListener("mousedown",this._events.clickOverlay),window.addEventListener("resize",this._events.resize),document.addEventListener("keydown",this._events.keyboardNav)}.call(this),document.body.insertBefore(this.modal,document.body.firstChild),this.opts.footer&&this.addFooter(),this},o.prototype._busy=function(o){t=o},o.prototype._isBusy=function(){return t},o.prototype.destroy=function(){null!==this.modal&&(this.isOpen()&&this.close(!0),function(){-1!==this.opts.closeMethods.indexOf("button")&&this.modalCloseBtn.removeEventListener("click",this._events.clickCloseBtn);this.modal.removeEventListener("mousedown",this._events.clickOverlay),window.removeEventListener("resize",this._events.resize),document.removeEventListener("keydown",this._events.keyboardNav)}.call(this),this.modal.parentNode.removeChild(this.modal),this.modal=null)},o.prototype.isOpen=function(){return!!this.modal.classList.contains("tingle-modal--visible")},o.prototype.open=function(){if(!this._isBusy()){this._busy(!0);return"function"==typeof this.opts.beforeOpen&&this.opts.beforeOpen(),this.modal.style.removeProperty?this.modal.style.removeProperty("display"):this.modal.style.removeAttribute("display"),this._scrollPosition=window.pageYOffset,document.body.classList.add("tingle-enabled"),document.body.style.top=-this._scrollPosition+"px",this.setStickyFooter(this.opts.stickyFooter),this.modal.classList.add("tingle-modal--visible"),"function"==typeof this.opts.onOpen&&this.opts.onOpen.call(this),this._busy(!1),this.checkOverflow(),this}},o.prototype.close=function(t){if(!this._isBusy()){if(this._busy(!0),t=t||!1,"function"==typeof this.opts.beforeClose)if(!this.opts.beforeClose.call(this))return void this._busy(!1);document.body.classList.remove("tingle-enabled"),window.scrollTo({top:this._scrollPosition,behavior:"instant"}),document.body.style.top=null,this.modal.classList.remove("tingle-modal--visible");this.modal.style.display="none","function"==typeof this.opts.onClose&&this.opts.onClose.call(this),this._busy(!1)}},o.prototype.setContent=function(t){return"string"==typeof t?this.modalBoxContent.innerHTML=t:(this.modalBoxContent.innerHTML="",this.modalBoxContent.appendChild(t)),this.isOpen()&&this.checkOverflow(),this},o.prototype.getContent=function(){return this.modalBoxContent},o.prototype.addFooter=function(){return function(){this.modalBoxFooter=document.createElement("div"),this.modalBoxFooter.classList.add("tingle-modal-box__footer"),this.modalBox.appendChild(this.modalBoxFooter)}.call(this),this},o.prototype.setFooterContent=function(t){return this.modalBoxFooter.innerHTML=t,this},o.prototype.getFooterContent=function(){return this.modalBoxFooter},o.prototype.setStickyFooter=function(t){return this.isOverflow()||(t=!1),t?this.modalBox.contains(this.modalBoxFooter)&&(this.modalBox.removeChild(this.modalBoxFooter),this.modal.appendChild(this.modalBoxFooter),this.modalBoxFooter.classList.add("tingle-modal-box__footer--sticky"),e.call(this),this.modalBoxContent.style["padding-bottom"]=this.modalBoxFooter.clientHeight+20+"px"):this.modalBoxFooter&&(this.modalBox.contains(this.modalBoxFooter)||(this.modal.removeChild(this.modalBoxFooter),this.modalBox.appendChild(this.modalBoxFooter),this.modalBoxFooter.style.width="auto",this.modalBoxFooter.style.left="",this.modalBoxContent.style["padding-bottom"]="",this.modalBoxFooter.classList.remove("tingle-modal-box__footer--sticky"))),this},o.prototype.addFooterBtn=function(t,o,e){var s=document.createElement("button");return s.innerHTML=t,s.addEventListener("click",e),"string"==typeof o&&o.length&&o.split(" ").forEach(function(t){s.classList.add(t)}),this.modalBoxFooter.appendChild(s),s},o.prototype.resize=function(){console.warn("Resize is deprecated and will be removed in version 1.0")},o.prototype.isOverflow=function(){var t=window.innerHeight;return this.modalBox.clientHeight>=t},o.prototype.checkOverflow=function(){this.modal.classList.contains("tingle-modal--visible")&&(this.isOverflow()?this.modal.classList.add("tingle-modal--overflow"):this.modal.classList.remove("tingle-modal--overflow"),!this.isOverflow()&&this.opts.stickyFooter?this.setStickyFooter(!1):this.isOverflow()&&this.opts.stickyFooter&&(e.call(this),this.setStickyFooter(!0)))},{modal:o}});

/* create a tingle modal dialog */ 
var tcc_modal = new tingle.modal({
  footer: false,
  stickyFooter: false,
  closeMethods: ['overlay', 'button', 'escape'],
  closeLabel: "Close",
  cssClass: ['custom-class-1', 'custom-class-2'],
  onOpen: function() {
     //console.log('modal open');
  },
  onClose: function() {
     //console.log('modal closed');
  },
  beforeClose: function() {
     return true; // close the modal
  }

  
});

/* create a SEARCH_AI tingle modal dialog */ 
var SEARCH_AI_modal = new tingle.modal({
   closeMethods: [],
   closeLabel: "Close",
   cssClass: ['custom-modal'],
   onOpen: function () {
      console.log('Modal opened');
   },
   onClose: function () {
      console.log('Modal closed');
   },
   beforeClose: function () {
      const isLoading = document.getElementById('SEARCH_AI_loading_overlay').style.display === 'block';
      return !isLoading; // prevent close during AI loading
   }
});




