window.onload = function() {
  var snowplowTracker = Snowplow.getTrackerUrl('sandstorm.elijah.cs.cmu.edu:8080');
  snowplowTracker.enableLinkTracking();
  snowplowTracker.trackPageView();
}
