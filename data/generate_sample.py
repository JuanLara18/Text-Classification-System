"""
Generate support_tickets_sample.csv — a synthetic customer support dataset
used in classifai examples and notebooks.

Run from the repo root:
    python data/generate_sample.py
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Template tickets per category
# ---------------------------------------------------------------------------

TICKETS = {
    "Billing & Payments": [
        ("Charged twice for my subscription", "Hi, I noticed two charges of $29.99 on my credit card for the same month. Could you please refund the duplicate charge? My account email is listed above."),
        ("Invoice shows wrong amount", "The invoice I received for last month shows $149 but my plan is $99/month. I have not added any extra services. Please correct this."),
        ("Cannot update payment method", "I am trying to update my credit card but the payment page keeps showing an error: 'Unable to save card'. I have tried three different browsers and it still fails."),
        ("Refund not received after 10 days", "You approved my refund on the 5th of this month (confirmation #RF-8821) but I have not seen the credit on my statement yet. My bank says no pending credit from you."),
        ("Unexpected charge after cancellation", "I cancelled my account two weeks ago and received a confirmation email, yet I was still charged this month. Please reverse this charge immediately."),
        ("Coupon code not applied at checkout", "I have a valid coupon code SAVE20 but when I enter it at checkout the total does not change. The code is supposed to give 20% off. Please help."),
        ("Need itemized receipt for expense report", "I need a detailed receipt for my purchase from last Tuesday for my company expense report. The amount was $79.00. Could you email me an itemized version?"),
        ("Annual plan price discrepancy", "Your website advertises the annual plan at $199/year but I was charged $239. I signed up using the link from your email campaign. Please honor the advertised price."),
        ("Auto-renewal turned on without my consent", "I did not authorize auto-renewal on my account. I see it was enabled last month and I was charged. I would like a refund and to turn off auto-renewal."),
        ("Bank says charge is from unknown merchant", "My bank flagged a charge from your company as potentially fraudulent because the merchant name on the statement does not match your company name. Can you provide the exact descriptor?"),
    ],
    "Technical Support": [
        ("App crashes on launch after update", "Since the update to version 3.2 yesterday, the app crashes immediately when I open it. I am on iPhone 14 running iOS 17. I have tried reinstalling but the problem persists."),
        ("Cannot connect to API — getting 401 errors", "My API integration stopped working this morning. All requests return HTTP 401 Unauthorized. My API key has not changed and was working fine yesterday."),
        ("Data export is stuck at 0%", "I started a data export 3 hours ago and the progress bar has not moved from 0%. The page just shows 'Preparing export...' I need this data urgently."),
        ("Sync not working between mobile and desktop", "My data on the mobile app is not syncing with the desktop version. I added 15 items on my phone today and none of them appear when I log in from my computer."),
        ("Login page keeps redirecting to itself", "When I try to log in I enter my credentials and the page just refreshes and shows the login form again. I never get into my account. I have cleared cache and cookies."),
        ("Search returns no results for any query", "The search feature has not been returning any results since yesterday afternoon. I searched for terms I know exist in my data and got 0 results every time."),
        ("Webhook not firing on new events", "I set up a webhook endpoint last week and it was working. As of Monday it is not receiving any POST requests despite new events occurring. The endpoint is responding correctly to pings."),
        ("PDF export renders tables incorrectly", "When I export reports to PDF, the tables are cut off at the right edge. The same report looks fine on screen. I have tried different paper sizes with the same result."),
        ("Two-factor authentication code not arriving", "I enabled 2FA on my account but the verification code is not arriving via SMS. I have waited 15 minutes and tried three times. My phone number is correct in my profile."),
        ("Bulk import fails with no error message", "I am trying to import a CSV file with 5,000 rows. The upload completes but then nothing happens — no error, no confirmation, no imported data. The file is formatted exactly as specified."),
    ],
    "Account & Access": [
        ("Cannot reset my password", "I clicked 'Forgot Password' and entered my email but I never receive the reset link. I have checked spam and waited an hour. My account email is the one I used to open this ticket."),
        ("Account locked after too many attempts", "My account appears to be locked. I tried to log in a few times with what I thought was my password and now I just see 'Account temporarily locked'. How long does this last?"),
        ("Need to change account email address", "I changed jobs and need to update the email on my account from my old work address to my personal email. I still have access to the old email to verify."),
        ("Teammate cannot accept invitation", "I sent an invitation to a colleague three days ago. She received the email but when she clicks the link it says 'Invitation expired or invalid'. Can you resend or generate a new link?"),
        ("Admin role not showing after upgrade", "I upgraded my account to the Business plan which should give me admin controls. However my dashboard looks the same as before and I do not see the admin panel."),
        ("SSO login not working with our identity provider", "We configured SAML SSO with our company's Okta instance following your documentation. When users try to log in via SSO they get a generic error page after being redirected back."),
        ("How do I transfer account ownership?", "Our original account owner has left the company. I am the new IT administrator and need to become the account owner. I have the original owner's work email if that helps."),
        ("Cannot access account after company domain change", "Our company rebranded and changed our email domain. I now cannot log into the account that was tied to my old email address. I need to access historical data there."),
        ("Multiple accounts merged without permission", "It looks like my personal and company accounts were merged. I now see data from both in one dashboard which is a serious privacy issue. Please separate them immediately."),
        ("Permission settings not saving", "I update the permission settings for a team member and click Save. The page confirms the change but when I reload the permissions are back to what they were before."),
    ],
    "Shipping & Delivery": [
        ("Package not delivered but marked as delivered", "My tracking shows the package was delivered yesterday but nothing arrived. My neighbor did not receive it either. Could you help locate it or start a claim?"),
        ("Wrong shipping address used on my order", "I noticed after placing the order that the wrong address was used — it went to my old address. The order number is #ORD-44291. Can this still be changed?"),
        ("Tracking number not working", "The tracking number you emailed me (1Z999AA10123456784) does not return any results on the carrier's website. I have been checking for two days."),
        ("Order shipped to wrong country", "My order was shipped to Germany but I am in Canada. I double-checked my account and my address shows Canada. This seems like a system error on your end."),
        ("Estimated delivery keeps changing", "The delivery date for my order has changed three times in the past week — first it was Tuesday, then Friday, now next Wednesday. Can you confirm when it will actually arrive?"),
        ("Package arrived damaged", "The box arrived today but it was clearly damaged in transit — one corner is completely crushed and the item inside is broken. I have taken photos. What is the process for a replacement?"),
        ("Never received shipping confirmation", "I placed an order five days ago. It was charged to my card but I never received a shipping confirmation or tracking number. My order status still shows 'Processing'."),
        ("Partial order received", "My order contained three items but only two arrived. The packing slip inside lists all three items. The missing item is the most important one I needed."),
        ("Shipment split into multiple packages unexpectedly", "I ordered one product but received a notification that it will be delivered in 4 separate packages. No one explained why. Will there be additional shipping charges?"),
        ("Delivery requires signature but I will not be home", "My tracking shows a signature is required for delivery but I will be at work. I tried to use the carrier's redirect tool but your shipping does not seem to allow it."),
    ],
    "Returns & Refunds": [
        ("How do I start a return?", "I received my order last week but it is not what I expected. I would like to return it. I could not find a return option in my account. What is the process?"),
        ("Return label not received", "I requested a return 5 days ago and was told a label would be emailed within 24 hours. I still have not received it. My return window closes in 3 days."),
        ("Refund issued for wrong amount", "I returned two items totaling $87. The refund posted to my card was $52. I need the remaining $35 refunded as well. My return confirmation is #RET-1190."),
        ("Item marked as non-returnable but it should not be", "The product page said 'Free returns within 30 days' but when I try to initiate a return it says this item is non-returnable. I purchased it 12 days ago."),
        ("Exchange instead of refund not processed", "I asked for an exchange for a different size, not a refund. The agent I chatted with confirmed an exchange. However I received a refund to my original payment and no replacement was sent."),
        ("Refund to wrong card", "My original card was cancelled and I got a new one with a different number. The refund was sent to the old cancelled card. How can we redirect it to my current card?"),
        ("Return window extended promise not honored", "A support agent told me via chat (I have the transcript) that my return window would be extended to 60 days due to a delay I experienced. Now the system is rejecting my return saying it is past 30 days."),
        ("Received wrong item and need to return it", "My order arrived today but the item inside is completely different from what I ordered. I obviously want to return this and get what I actually paid for."),
    ],
    "Product Information": [
        ("Does your product support SAML SSO?", "We are evaluating your product for our enterprise. Before we proceed, I need to confirm: does the Business plan support SAML 2.0 SSO with major identity providers like Okta and Azure AD?"),
        ("What is the data retention policy?", "For compliance reasons I need to understand how long you retain customer data after account deletion and whether we can request full data deletion under GDPR."),
        ("Is there an offline mode?", "I often work in locations with poor internet connectivity. Is there a way to use the product offline and sync when connectivity is restored?"),
        ("What file formats can I import?", "I have data in multiple formats: CSV, Excel, and JSON. Can your import feature handle all three? Are there any row or file size limits I should be aware of?"),
        ("Does the API support bulk operations?", "I need to update thousands of records at once. Does your API have batch endpoints or would I need to make individual calls for each record? What are the rate limits?"),
        ("Compatibility with Windows 11", "Before purchasing I want to confirm the desktop application is fully compatible with Windows 11. Are there any known issues with the latest Windows update?"),
        ("Maximum number of team members on Starter plan", "Your pricing page mentions 'up to 5 team members' on the Starter plan. Does that include the account owner or is it 5 additional members on top of the owner?"),
        ("Can reports be scheduled for automatic delivery?", "I want to set up weekly reports to be automatically emailed to our leadership team every Monday morning. Is this a built-in feature or would I need to use the API?"),
    ],
}

# ---------------------------------------------------------------------------
# Generate rows
# ---------------------------------------------------------------------------

def random_date(start_days_ago=90):
    delta = timedelta(days=random.randint(0, start_days_ago),
                      hours=random.randint(0, 23),
                      minutes=random.randint(0, 59))
    return (datetime.now() - delta).strftime("%Y-%m-%d %H:%M")

rows = []
ticket_id = 1000

for category, examples in TICKETS.items():
    for subject, body in examples:
        rows.append({
            "id": f"TKT-{ticket_id}",
            "created_at": random_date(),
            "subject": subject,
            "body": body,
            "true_category": category,   # ground truth — use to evaluate accuracy
        })
        ticket_id += 1

random.shuffle(rows)

# Re-assign IDs after shuffle
for i, row in enumerate(rows):
    row["id"] = f"TKT-{1000 + i}"

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

out = Path(__file__).parent / "support_tickets_sample.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "created_at", "subject", "body", "true_category"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} tickets -> {out}")
