// Research-only fresh-circuit CPU runner. NO precompiled circuit plans or
// numerical result cache: each invocation reads and validates a supplied
// quantized IBM Sherbrooke-style calibration/connectivity snapshot, constructs
// the same exact gate/noise stage sequence and causal cones, then calls the
// unchanged run_pauli_indexed numerical kernel in compiled_plan_runner.cpp.
// Restricted scope: 127 qubits; 2/4 layers; SX, RZ, ECR with Aer-style
// depolarization, T1/T2 and asymmetric Z-readout. Other algorithms refused.
#define main precompiled_runner_original_main
#include "compiled_plan_runner.cpp"
#undef main
#include <sstream>
#include <map>
#include <set>
#include <iterator>
#include <numeric>

struct CalibrationQ {double t1, t2, p01, p10, sxerr;};
struct CalibrationE {int a,b;double err,t;};
struct Calibration {std::vector<CalibrationQ> q;std::vector<CalibrationE> e;};

static uint64_t b36(const std::string& s){
 if(s.empty()||s.size()>16)throw std::runtime_error("invalid compact numeric field");
 uint64_t value=0;
 for(char c:s){int k=c>='0'&&c<='9'?c-'0':c>='a'&&c<='z'?c-'a'+10:-1;
   if(k<0||value>(UINT64_MAX-static_cast<uint64_t>(k))/36)
      throw std::runtime_error("invalid compact base36 number");
   value=value*36+static_cast<uint64_t>(k);
 }
 return value;
}
static std::vector<std::string> split(const std::string& s,char c){
 std::vector<std::string> out;size_t start=0;
 while(true){auto i=s.find(c,start);out.push_back(s.substr(start,i==std::string::npos?i:i-start));
  if(i==std::string::npos)break;start=i+1;
 }return out;
}
static std::vector<std::vector<uint64_t>> compact_records(const std::string& data,int nf,int n){
 auto parts=split(data,'.');if(static_cast<int>(parts.size())!=n)throw std::runtime_error("wrong compact row count");
 std::vector<std::vector<uint64_t>> out;out.reserve(n);
 for(auto const& p:parts){auto fields=split(p,'_');if(static_cast<int>(fields.size())!=nf)throw std::runtime_error("wrong row format");
   std::vector<uint64_t> row;for(auto const& f:fields)row.push_back(b36(f));out.push_back(std::move(row));
 }return out;
}
static Calibration read_calibration(const std::string& file){
 std::ifstream f(file,std::ios::binary|std::ios::ate);
 if(!f)throw std::runtime_error("missing circuit snapshot");
 auto n=f.tellg();
 if(n<=0||n>32768)throw std::runtime_error("invalid snapshot size");
 std::string s(static_cast<size_t>(n),'\0');
 f.seekg(0);
 if(!f.read(s.data(),n))throw std::runtime_error("snapshot read failed");
 std::map<std::string,std::string> kv;
 for(auto const& line:split(s,'\n')){
   if(line.empty())continue;auto pos=line.find('=');
   if(pos==std::string::npos||!kv.emplace(line.substr(0,pos),line.substr(pos+1)).second)
      throw std::runtime_error("invalid snapshot record");
 }
 if(kv.size()!=2||!kv.count("QROWS")||!kv.count("EROWS"))
   throw std::runtime_error("unsupported snapshot format");
 auto qrows=compact_records(kv.at("QROWS"),5,127);
 auto erows=compact_records(kv.at("EROWS"),4,144);
 Calibration cal;cal.q.reserve(127);cal.e.reserve(144);
 for(auto const& v:qrows){
   CalibrationQ q{double(v[0])*1e-9,double(v[1])*1e-9,double(v[2])/2048.,
     double(v[3])/2048.,double(v[4])*1e-9};
   if(!(q.t1>0&&q.t2>0&&q.t2<=2*q.t1*(1+1e-12)&&q.p01>=0&&q.p01<=1&&
       q.p10>=0&&q.p10<=1&&q.sxerr>=0&&q.sxerr<=1))throw std::runtime_error("invalid calibrated qubit");
   cal.q.push_back(q);
 }
 std::set<std::pair<int,int>> seen;
 for(auto const& v:erows){
   if(v[0]>=127||v[1]>=127||v[0]==v[1]||!seen.emplace(int(v[0]),int(v[1])).second)
     throw std::runtime_error("invalid or duplicate directed ECR coupling");
   CalibrationE e{int(v[0]),int(v[1]),double(v[2])*1e-7,double(v[3])*1e-12};
   if(!(e.err>=0&&e.err<=1&&e.t>=0&&e.t<=1))throw std::runtime_error("invalid calibrated coupler");
   cal.e.push_back(e);
 }
 std::sort(cal.e.begin(),cal.e.end(),[](auto const& a,auto const& b){return std::tie(a.a,a.b)<std::tie(b.a,b.b);});
 return cal;
}
// Zero-temperature Aer 0.18.2 gate-error contract. First deduct the
// infidelity explained by relaxation, then translate residual depolarization
// to the total probability of a nonidentity Pauli event.
static double calibrated_p(double err,double duration,const CalibrationQ& a,const CalibrationQ* b=nullptr){
 int dim=b?4:2;
 auto trace_one=[&](auto const& q){return 1+2*std::exp(-duration/q.t2)+std::exp(-duration/q.t1);};
 double tr=trace_one(a)*(b?trace_one(*b):1.);
 double fid=(tr+dim)/(dim*(dim+1.));
 double e=std::min(err,dim/(dim+1.));
 double pdep=0.;
 if(e>1-fid){pdep=dim*(e-(1-fid))/(dim*fid-1);pdep=std::min(std::max(0.,pdep),dim*dim/(dim*dim-1.));}
 return pdep*(1-1./(dim*dim));
}
static void put_ptm(CStage& s,int arity,const std::vector<std::pair<int,double>>& entries){
 int d=arity==1?4:16;
 for(auto e:entries){if(e.first<0||e.first>=d*d)throw std::runtime_error("invalid PTM");s.ptm[e.first]=e.second;}
}
static CStage make_stage(int a,int b,int param=-1){CStage s{};s.q0=a;s.q1=b;s.arity=(b<0?1:2);s.param=param;return s;}
static void add_sx(Plan& p,int q){
 auto s=make_stage(q,-1);
 put_ptm(s,1,{{0,1.},{5,1.},{11,1.},{14,-1.}});
 p.stages.push_back(s);p.bases.push_back(std::numeric_limits<double>::quiet_NaN());
}
static void add_ecr(Plan& p,int q0,int q1){
 auto s=make_stage(q0,q1);
 // Exact signed-permutation transfer matrix from U_ECR^dag P U_ECR.
 // P basis order (I,X,Z,Y) and wire0 as least-significant local Pauli.
 static const int cols[16]={0,1,11,10,13,12,6,7,8,9,3,2,5,4,14,15};
 static const int signs[16]={1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1};
 for(int row=0;row<16;row++)s.ptm[16*row+cols[row]]=signs[row];
 p.stages.push_back(s);p.bases.push_back(std::numeric_limits<double>::quiet_NaN());
}
static void add_depol(Plan& p,int q0,int q1,double pauli_probability){
 auto s=make_stage(q0,q1);int dim=q1<0?4:16;
 double scale=q1<0?1-4*pauli_probability/3:1-16*pauli_probability/15.;
 for(int k=0;k<dim;k++)s.ptm[k*dim+k]=(k==0?1.:scale);
 p.stages.push_back(s);p.bases.push_back(std::numeric_limits<double>::quiet_NaN());
}
static void add_relax(Plan& p,int q,double t,CalibrationQ const& qc){
 auto s=make_stage(q,-1);double l=std::exp(-t/qc.t1),trans=std::exp(-t/qc.t2);
 s.ptm[0]=1;s.ptm[5]=trans;s.ptm[8]=1-l;s.ptm[10]=l;s.ptm[15]=trans;
 p.stages.push_back(s);p.bases.push_back(std::numeric_limits<double>::quiet_NaN());
}
static void add_rz(Plan& p,int q,int param,double base){
 auto s=make_stage(q,-1,param);p.stages.push_back(s);p.bases.push_back(base);
}
struct Mask {uint64_t lo=0,hi=0;
 void set(int index){if(index<64)lo|=1ULL<<index;else hi|=1ULL<<(index-64);}
 Mask operator|(Mask other)const{return {lo|other.lo,hi|other.hi};}
};
static Plan construct_fresh(Calibration const& cal,int depth,int max_terms){
 Plan p;p.nqubits=127;p.nparams=depth;p.max_terms=max_terms;
 p.stages.reserve(depth*(127*4+144*4));p.bases.reserve(depth*(127*4+144*4));
 for(int layer=0;layer<depth;layer++){
   for(int q=0;q<127;q++){
      auto const& qc=cal.q[q];
      add_sx(p,q);
      constexpr double sxt=56.888888888888886e-9;
      add_depol(p,q,-1,calibrated_p(qc.sxerr,sxt,qc));
      add_relax(p,q,sxt,qc);
      add_rz(p,q,layer,.27+.04*(layer%4));
   }
   int start=(layer*43)%static_cast<int>(cal.e.size());bool used[127]{};
   for(int i=0;i<static_cast<int>(cal.e.size());i++){
      auto const& e=cal.e[(start+i)%cal.e.size()];
      if(used[e.a]||used[e.b])continue;
      used[e.a]=used[e.b]=true;
      add_ecr(p,e.a,e.b);
      add_depol(p,e.a,e.b,calibrated_p(e.err,e.t,cal.q[e.a],&cal.q[e.b]));
      add_relax(p,e.a,e.t,cal.q[e.a]);add_relax(p,e.b,e.t,cal.q[e.b]);
   }
 }
 if(p.stages.empty()||p.stages.size()>MAX_STAGES)throw std::runtime_error("unsupported stage count");
 // One exact reverse traversal for all 127 local observables.
 std::vector<Mask> query_mask(127),membership(p.stages.size());
 p.cones.resize(127);
 for(int q=0;q<127;q++){
   auto const& c=cal.q[q];double alpha=c.p10-c.p01,beta=1-c.p01-c.p10;
   auto& cone=p.cones[q];
   if(alpha!=0.){CTerm term{};term.coef=alpha;cone.terms.push_back(term);}
   if(beta!=0.){CTerm term{};term.coef=beta;term.z[q>>6]=1ULL<<(q&63);cone.terms.push_back(term);query_mask[q].set(q);}
   if(cone.terms.empty())throw std::runtime_error("zero measurement observable");
 }
 for(size_t k=p.stages.size();k-- >0;){
   auto const& st=p.stages[k];Mask active=query_mask[st.q0];
   if(st.arity==2){active=active|query_mask[st.q1];query_mask[st.q1]=active;}
   query_mask[st.q0]=active;membership[k]=active;
 }
 for(size_t k=0;k<p.stages.size();k++){
   auto m=membership[k];
   while(m.lo){int bit=__builtin_ctzll(m.lo);p.cones[bit].indices.push_back(static_cast<int32_t>(k));m.lo&=m.lo-1;}
   while(m.hi){int bit=__builtin_ctzll(m.hi);p.cones[bit+64].indices.push_back(static_cast<int32_t>(k));m.hi&=m.hi-1;}
 }
 for(auto const& m:query_mask){auto a=m;
   while(a.lo){int bit=__builtin_ctzll(a.lo);p.cones[bit].wire_count++;a.lo&=a.lo-1;}
   while(a.hi){int bit=__builtin_ctzll(a.hi);p.cones[bit+64].wire_count++;a.hi&=a.hi-1;}
 }
 return p;
}
int main(int argc,char**argv){
 try{
  if(argc<3||argc>5)throw std::runtime_error("usage: fresh_circuit_runner SNAPSHOT.txt DEPTH:2|4 [threads:1..4] [max_requests:1..256]");
  std::string file=argv[1];int depth=std::stoi(argv[2]);
  int threads=argc>3?std::stoi(argv[3]):1,maxreq=argc>4?std::stoi(argv[4]):1;
  if((depth!=2&&depth!=4)||threads<1||threads>4||maxreq<1||maxreq>256)
    throw std::runtime_error("unsupported circuit or runtime settings");
  auto start=Clock::now();auto cal=read_calibration(file);double parse_ms=elapsed_ms(start);
  start=Clock::now();Plan plan=construct_fresh(cal,depth,15000);double build_ms=elapsed_ms(start);
  std::cout<<"{\"kind\":\"ready\",\"nqubits\":127,\"snapshot_parse_ms\":"<<parse_ms
    <<",\"model_and_cones_ms\":"<<build_ms<<",\"stages\":"<<plan.stages.size()<<",\"queries\":"<<plan.cones.size()<<"}\n"<<std::flush;
  std::string line;int count=0;
  while(count<maxreq&&std::getline(std::cin,line)){
    if(line.size()>1024*1024){std::cout<<"{\"kind\":\"error\",\"ok\":false,\"error\":\"input limit\"}\n"<<std::flush;break;}
    try{process(plan,line,threads);}catch(std::exception const&e){
      std::cout<<"{\"kind\":\"error\",\"ok\":false,\"error\":\"request rejected\"}\n"<<std::flush;
      std::cerr<<"request error: "<<e.what()<<'\n';
    }++count;
  }
  return 0;
 }catch(std::exception const&e){std::cerr<<"fresh circuit failure: "<<e.what()<<'\n';return 2;}
}
