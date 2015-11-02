window.onload = function() {
  var snowplowTracker = Snowplow.getTrackerUrl('derecho.elijah.cs.cmu.edu:8080');
  snowplowTracker.enableLinkTracking();
  snowplowTracker.trackPageView();
}
