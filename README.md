# hikepnw
## What is hikepnw?
A [web app](http://hikepnw.us/) to order up personalized hiking trail recommendations in the Pacific Northwest using data science.

## How do you use hikepnw?
The Pacific Northwest offers a myriad of hiking options - from the beaches to the mountains to the deserts and even some urban adventures, it can almost be overwhelming to determine where to hike next. That's where hikepnw comes in. By leaning on the expertise of scores of seasoned hikers, hikepnw provides hiking trail recommendations that are tailored to your specific interests. Simply indicate a general location, describe your ideal hike in a few key words, and apply any "must have" filters, and you'll receive up to 10 hikes that you're likely gonna love.

## Data Engineering
The data behind hikepnw.us consists of trail information, trail ratings, and trail reviews from AllTrails.com®. More than 150,000 reviews from over 60,000 users are harnessed to make expert recommendations across over 3,200 hiking trails tailored to the specific interests of individual users.

To amass the data needed to make hiking trail recommendations, data from AllTrails.com® was scraped using selenium, beautiful soup, and sqlalchemy into a postgreSQL database, transformed and cleaned using pandas, numpy, and other python tools, and exported to nice clean csv files used for recommendation system model training. These files were then migrated to an ec2 instance where they were used to create a mini-data warehouse on the ec2 instance. The web app pulls data from the ec2 postgreSQL mini-data warehouse. 

## Data Science
Hiking trail recommendations are determined using a hybrid recommendation algorithm called LightFM. Using this hybrid model, latent embedding for new and existing users are learned in concert with user representations, collaborative filtering from user ratings, and a Weighted Approximate-Rank Pairwise loss function through stochastic gradient descent. User representations are determined by extracting key trail-descriptors from user reviews. The bridging principal is that users old and new who share similar vocabularies and who use similar descriptors are more likely to desire similar outdoor experiences. For example, some users are looking for 'epic' hikes that are 'cool' and 'challenging', while others might be looking for hikes that are 'peaceful', 'relaxing', and 'family friendly'.

To help refine recommendations, a couple of "must have" trail filters are applied based on trail features. This makes sure that recommended trails are truly personalized to the interests of the user.
