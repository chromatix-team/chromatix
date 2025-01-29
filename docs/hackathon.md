# Hackathon

We’re happy to share that we’re organizing the third edition of our annual Chromatix hackathon, this year happening **May 27rd – 31st at Janelia Research campus**! Many of you will probably be familiar with Chromatix, but for those that aren’t: Chromatix is a fully-differentiable, GPU-accelerated wave-optics library built with Jax we developed for several of our internal projects. The previous hackathons were a big success that we decided to organize another – and you or anyone from your group is invited to apply!

## What we’ll be doing
The concept is the same as usual: we’ll gather a group of around 15-20 people working on computational optics, and together we’ll implement various setups or new ideas in Chromatix. At the end of the week, participants have typically added a new feature or translated their setup into Chromatix. For example, projects have included 3D CGH, estimating system aberrations via phase retrieval and advanced propagation methods. During the entire hackathon the two original developers (Diptodip and Gert-Jan) will be around so you can focus on your idea instead of being stuck on programming issues. 
We’ll pair you with someone working on a similar idea, and together you’ll work on making a prototype. The first day will be an introduction to Chromatix, followed by three days of project work, and finally presentations the last morning. This might not sound very long, but we’ve seen some very impressive results!

Above all the hackathon is also a social experience – you'll be working and solving problems together with people from various backgrounds and institutions, making it also an excellent opportunity to build your network.

## What we expect from you
A project to work on! This can be anything from translating an existing piece of code which you want to make cleaner and run on GPUs, to implementing an entirely new optical system or idea. We have a list of [ideas](#Inspiration) if you're looking for inspiration.

We don’t need you to be a programming wizard, but we do ask for basic programming skills.

## Applying
Excited? To apply, send an email to bothg@hhmi.org with the following information:

1. A description of what you want to do – about half a page to a page. 
2. Your resume.

Please combine them into a single pdf, and use ‘Application Chromatix Hackathon’ as subject. You can apply until the 16th of March and we’ll let you know the 21st of March if you’re accepted. You’ll be housed on our campus - so you’ll only need to cover transport costs. Please contact us if this is a problem and we’ll see if we can arrange something. 


## Inspiration

Here's a list of things which are either not implemented yet or interest us:

* Super resolution setups: we haven't modelled super-resolution microscopes such as STED yet.
* Modified Born Series: The MBS is an exciting approach to exactly solve Maxwell's equations. How much can it improve sample reconstruction?
* High-NA objectives: Most modern microscopes use high-NA objectives. Can we model these, and how much does it change the simulation?
* LLMs to build setups: Can we use an LLM to read a paper, and automatically construct the corresponding Chromatix setup? 
* Extremely large reconstructions: how far can we push the envelope on reconstruction size using Chromatix?
* Implicit neural representations: how well can NERFs replace voxelgrids?
* Can we use LLMs to build and expand documentation? 




