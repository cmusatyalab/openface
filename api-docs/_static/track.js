$(document).ready(function() {
    try {
        var snowplowTracker = Snowplow.getTrackerUrl('joule.isr.cs.cmu.edu:8081');
        snowplowTracker.enableLinkTracking();
        snowplowTracker.trackPageView();
    } catch (err) {}
});
