# matchup-generator
Tool to determine optimal pitch sequencing based on hitter/pitcher matchups

Try it at http://matchup-generator.streamlit.app/

## Note

Currently, this app is heavily limited by the lack of context for each individual pitch: i.e., each pitch is viewed in a vacuum based on the hitter, pitcher, and count, when in real life a pitch's effectiveness is heavily tied to how well it plays off other pitches. Adding this feature is a work in progress, with variables like previous pitch type, previous pitch location, times a specific pitch has been thrown in an AB, the pitch number of an AB, and how many times the pitcher has gone through the lineup set to be added.

## Future Updates

Best and Worst: Given a user-inputted batter, generate which pitchers' arsenals project to be most and least effective against him, and vice-versa

Pitcher recommendations for multiple batters: Allows the user to select a lineup of hitters, filter the pitching team, and see which pitcher matches up best
