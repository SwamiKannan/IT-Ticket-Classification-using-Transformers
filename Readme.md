# IT Ticket Classification using Transformers
<p align="center">
<img src="https://github.com/SwamiKannan/IT-Ticket-Classification-using-Transformers/blob/main/cover.png" width=50% height=50%>
</p>

## Data:
One of the key activities of any IT function is to “Keep the lights on” to ensure there is no impact to the Business operations. IT leverages Incident Management process to achieve the above Objective. An incident is something that is unplanned interruption to an IT service or reduction in the quality of an IT service that affects the Users and the Business. The main goal of Incident Management process is to provide a quick fix / workarounds or solutions that resolves the interruption and restores the service to its full capacity to ensure no business impact.<br><br>
Currently the incidents are created by various stakeholders (Business Users, IT Users and Monitoring Tools) within IT Service Management Tool and are assigned to Service Desk teams (L1 / L2 teams). This team will review the incidents for right ticket categorization, priorities and then carry out initial diagnosis to see if they can resolve. Around ~54% of the incidents are resolved by L1 / L2 teams. In case L1 / L2 is unable to resolve, they will then escalate / assign the tickets to Functional teams from Applications and Infrastructure (L3 teams). Some portions of incidents are directly assigned to L3 teams by monitoring tools or Callers /Requestors. L3 teams will carry out detailed diagnosis and resolve the incidents. Around ~56% of incidents are resolved by Functional / L3 teams. In case if vendor support is needed, they will reach out for their support towards incident closure<br>
## Data Reference:
https://github.com/SwamiKannan/IT-Ticket-Classification-using-Transformers/blob/main/DATA/RAW_DATA/Input%20Data%20Translated.csv

## Key Ask:<br>
<ul>
	<li> There are 74 different classes in the original dataset ! Many of these had only one ticket mapped to the class !</li>
	<li> One of the classes constituted nearly 46% of the total trouble tickets. This dataset was massively unbalanced </li>
	<li> There were only 80,000 samples for analysis (or an average of just 1,081 per class) which is relatively low for a transformer  
  <li>Guided by powerful AI techniques that can classify incidents to right functional groups can help organizations to reduce the resolving time of the issue and can focus on more productive tasks.</li>
<li>The goal is to build a Transformer-based classifier that can classify the tickets by analysing text.</li>
<br>
<sub> Image credit: https://www.itarian.com/images/free-online-ticketing-system.png </sub>
