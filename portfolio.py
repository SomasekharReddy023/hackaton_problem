import requests
from bs4 import BeautifulSoup

# ------------------ LEETCODE ------------------
def fetch_leetcode(username):
    try:
        url = "https://leetcode.com/graphql"
        query = {
            "query": """
            query getUserProfile($username: String!) {
              matchedUser(username: $username) {
                username
                submitStats: submitStatsGlobal {
                  acSubmissionNum {
                    difficulty
                    count
                  }
                }
              }
            }
            """,
            "variables": {"username": username}
        }
        r = requests.post(url, json=query, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()

        total = data["data"]["matchedUser"]["submitStats"]["acSubmissionNum"][0]["count"]

        return {
            "platform": "LeetCode",
            "username": username,
            "problems_solved": total,
            "level": "Intermediate" if total > 100 else "Beginner"
        }
    except Exception as e:
        return {"platform": "LeetCode", "username": username, "error": str(e)}


# ------------------ CODECHEF ------------------
def fetch_codechef(username):
    try:
        url = f"https://www.codechef.com/users/{username}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        rating_tag = soup.find("div", class_="rating-number")
        rating = rating_tag.text.strip() if rating_tag else "N/A"

        return {
            "platform": "CodeChef",
            "username": username,
            "rating": rating,
            "level": "Intermediate" if rating != "N/A" and rating.isdigit() and int(rating) > 1500 else "Beginner"
        }
    except Exception as e:
        return {"platform": "CodeChef", "username": username, "error": str(e)}


# ------------------ MAIN ------------------
if __name__ == "__main__":
    print("⚡ Portfolio Stats Fetcher Running...\n")

    leet = fetch_leetcode("venkataramanaa")
    print("🔎", leet)

    chef = fetch_codechef("venkataramanaa")
    print("🔎", chef)

