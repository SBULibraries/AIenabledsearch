

(function () {
   "use strict";
   'use strict';

   var app = angular.module('viewCustom', ['angularLoad']);

   /* JQUERY */
   /* This code adds jQuery, which is required for some customizations */
   var jQueryScript = document.createElement("script");
   jQueryScript.src = "https://code.jquery.com/jquery-3.3.1.min.js";
   document.getElementsByTagName("head")[0].appendChild(jQueryScript);

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




