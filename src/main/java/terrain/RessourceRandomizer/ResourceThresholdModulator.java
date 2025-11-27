package terrain.RessourceRandomizer;

// Manages terrain-aware threshold modulation for resource placement
// Different resources have different biome preferences and flatness requirements
public class ResourceThresholdModulator {


    /* Terrain Types:
    0 : WATER
    2 : SHORELINE
    3 : PLAINS
    4 : FOOTHILLS
    5 : HILLS
    6 : MOUNTAIN_BASE
    7 : MOUNTAINS
    */

    /* Resource Types:
    0 : SEDIMENTARY_ROCK 
    1 : GEMSTONES 
    2 : IRON_ORE 
    3 : COAL 
    4 : GOLD_ORE 
    5 : WOOD 
    6 : CATTLE_HERD 
    7 : WOLF_PACK 
    8 : FISH_SCHOOL 
    */

    public static class ResourcePreferences {
        public final int favorableTerrain1;
        public final int favorableTerrain2;
        public final double favorableTerrainBonus;
        public final int unfavorableTerrain;
        public final double unfavorablePenalty;
        public final double flatnessPreference; 
        public final double flatnessInfluence;

        public ResourcePreferences(int fav1, int fav2, double favBonus,
                                   int unfav, double unfavPenalty, double flatPref, double flatInfluence) {
            this.favorableTerrain1 = fav1;
            this.favorableTerrain2 = fav2;
            this.favorableTerrainBonus = favBonus;
            this.unfavorableTerrain = unfav;
            this.unfavorablePenalty = unfavPenalty;
            this.flatnessPreference = flatPref;
            this.flatnessInfluence = flatInfluence;
        }
    }

    // Define preferences for each resource type
    private static final ResourcePreferences[] preferences =  {
        // SEDIMENTARY_ROCK: hugs coasts and flood plains, avoids high peaks
        new ResourcePreferences(2, 3, 0.05, 7, 0.18, 0.75, 0.25),

        // GEMSTONES: embedded in rugged alpine biomes with steep slopes
        new ResourcePreferences(6, 7, 0.04, 2, 0.24, 0.25, 0.35),

        // IRON_ORE: concentrated in foothills and hill belts, shuns wetlands
        new ResourcePreferences(4, 5, 0.06, 0, 0.18, 0.45, 0.20),

        // COAL: forms beneath gently rolling plains and foothills
        new ResourcePreferences(3, 4, 0.05, 0, 0.17, 0.70, 0.22),

        // GOLD_ORE: rare pockets in the harshest mountains
        new ResourcePreferences(6, 7, 0.06, 3, 0.28, 0.30, 0.30),

        // WOOD: thrives in broadleaf plains spilling into foothills
        new ResourcePreferences(3, 4, 0.04, 7, 0.22, 0.80, 0.18),

        // CATTLE_HERD: needs expansive flatlands and meadow edges
        new ResourcePreferences(3, 4, 0.05, 5, 0.25, 0.90, 0.40),

        // WOLF_PACK: roams forest fringes between plains and foothills
        new ResourcePreferences(4, 5, 0.03, 0, 0.20, 0.50, 0.15),

        // FISH_SCHOOL: confined to deep and shallow waters, penalized elsewhere
        new ResourcePreferences(0, 2, 0.08, 3, 0.30, 0.95, 0.10)
    };

    public static double modulateThreshold(int resourceType, int terrain, double flatness, double threshold) {
        if (resourceType < 0 || resourceType >= preferences.length) {
            return threshold;
        }

        ResourcePreferences pref = preferences[resourceType];

        // Adjust based on terrain type
        if (terrain == pref.favorableTerrain1 || terrain == pref.favorableTerrain2) {
            threshold -= pref.favorableTerrainBonus;
        } else if (terrain == pref.unfavorableTerrain) {
            threshold += pref.unfavorablePenalty;
        }

        // Adjust based on flatness preference
        double flatnessDifference = Math.abs(flatness - pref.flatnessPreference);
        threshold += flatnessDifference * pref.flatnessInfluence;

        return threshold;
    }
}
