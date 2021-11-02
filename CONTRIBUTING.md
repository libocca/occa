# Get Involved

**OCCA is a community driven project that relies on the support of people like you. Thank you!** 

## Dip Your Toes

### Report Bugs

### Suggest enhancements

- *How can we make OCCA better?*
- *What features do our users need?*
- *What new features are common to backend programming models that should be incorporated into OCCA*?

## Wade-in

### User documentation

### Testing

## Dive Deep

### Bug fixes

- Be sure to tag open issue related to the bug.

### New features

*Thoughts:* 
 - *Should open an issue first so there is opportunity for open discussion.*
 - 
### Developer documentation

# Best practices

*Introductory sentence about purpose for these. E.g., we want to create a culture of apprenticeship&mdash;where people can learn and "make mistakes"&mdash;while still maintaining high standards for our software.*

## Issues

- *Issues templates for*
  - *bugs*
  - *enhancements*
  - *documentation for*
    - *users*
    - *developers*

## Pull-requests

- *Open these "early" (i.e. during development) as a draft. This will allow for feedback during development, facilitate collaboration, and reduce turnaround time for reviews.
- Clearly document PR so we can include this info in the notes for the next release

### Better together

- *How to contribute to an open pull-request*
- *How to seek collaborators*

### Automated testing

When a pull-request is submitted, a series of automated tests is triggered. New code must pass all tests before it will be integrated. Details on OCCA's automated testing can be found in the [developer documentation]().

> **Important**: Pull-requests for new feautures commonly pass all unit and integration tests only to fail the code coverage requirements. Tests for new code should be written and added to the CTest framework as part of the development process. See the [developer documentation]() for details.

### Code review

*Be specific about*:
- *Timing to expect review*
- *What we are looking for*
- *Request for review(?)*

### Integration

The admin team will aim to merge new code which has been reviewed and passed automated testing with **XXX** business days.

### Release

*For admin team:*

 - *What should determine our release cadence?*
    - Fixed time intervals (e.g. quarterly, semiannually). 
      - Default to including all PRs since last release.
    - A specific set of issues.
      - Create version milestones and gh project board to help highlight best areas for core developers to focus their efforts

<!--
Does GitHub have a way to automate generation the initial draft of release notes? Otherwise, we should write a script to do this and setup a new gh action.
--->